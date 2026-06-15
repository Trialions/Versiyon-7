# walk_forward.py — Aylık Robustluk / Walk-Forward Benzeri Analiz Aracı v3
#
# NOT:
#   Bu dosya klasik Train → Optimize → Test walk-forward değildir.
#   Aynı config'i aylık parçalara bölerek test eder. Ama v2'ye göre kritik
#   doğruluk düzeltmeleri içerir:
#
#   1) Warmup eklendi:
#      Her ay, indikatör/HTF/BTC buffer'ları önceki mumlarla ısıtılır.
#      Warmup döneminde işlem açılmaz. Böylece ayın ilk günleri 50 mum / HTF
#      eksikliği yüzünden yapay olarak filtrelenmez.
#
#   2) HTF warmup düzeltildi:
#      4h HTF buffer'ı segment başlangıcından önceki HTF mumlarıyla doldurulur,
#      test içinde sadece timestamp <= mevcut mum olan HTF mumları eklenir.
#      Gelecek HTF verisi kullanılmaz.
#
#   3) BTC filter warmup düzeltildi:
#      BTCUSDT geçmiş kapanışları test ayı başlamadan önce buffer'a eklenir.
#
#   4) Metrikler genişletildi:
#      PF, avg win/loss, RR, recovery, max ardışık kayıp, exit reason sayıları,
#      en iyi/en kötü sembol aylık ve genel rapora eklenir.
#
#   5) Universe uyarısı eklendi:
#      symbols_top70.json bugünün listesi ise geçmiş ay testlerinde look-ahead bias
#      oluşturabilir. Bu nedenle çıktı summary içine universe_mode yazılır.
#
# Kullanım:
#   python walk_forward.py --start 2025-01-01 --end 2026-01-01 --interval 1h --top 20 --out walk_forward_results
#   python walk_forward.py --start 2025-01-01 --end 2026-01-01 --symbols BTCUSDT,ETHUSDT,ZECUSDT --out walk_forward_results
#   python walk_forward.py --start 2025-01-01 --end 2026-01-01 --top 20 --weekly-symbols --symbol-lookback-days 30 --out walk_forward_results
#
# Öneri:
#   Bu aracı "Aylık Robustluk Testi" olarak yorumla. Gerçek WF istenirse ayrı
#   optimizer entegrasyonlu Train→Optimize→Test döngüsü yazılmalı.

from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple, Any

import yaml

import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))

from backtest import (
    Backtester,
    fetch_klines,
    fetch_funding_rates,
    _load_cache,
    _save_cache,
    _max_drawdown,
    _sharpe,
    generate_report,
)


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _to_ms_utc(dt: datetime) -> int:
    """Naive YYYY-MM-DD tarihini UTC kabul ederek ms timestamp döndürür."""
    return int(calendar.timegm(dt.timetuple()) * 1000)


def _date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _month_ranges(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """Başlangıç-bitiş arasını aylık test dilimlerine böler. end exclusive."""
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end <= start:
        raise ValueError("end tarihi start tarihinden büyük olmalı")

    ranges = []
    cur = start
    while cur < end:
        if cur.month == 12:
            nxt = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            nxt = cur.replace(month=cur.month + 1, day=1)
        seg_end = min(nxt, end)
        ranges.append((_date_str(cur), _date_str(seg_end)))
        cur = nxt
    return ranges




def _week_ranges(start_date: str, end_date: str, refresh_days: int = 7) -> List[Tuple[str, str]]:
    """Başlangıç-bitiş arasını haftalık/seçilen gün sayılı dilimlere böler. end exclusive."""
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end <= start:
        raise ValueError("end tarihi start tarihinden büyük olmalı")
    ranges = []
    cur = start
    step = timedelta(days=max(1, int(refresh_days)))
    while cur < end:
        nxt = min(cur + step, end)
        ranges.append((_date_str(cur), _date_str(nxt)))
        cur = nxt
    return ranges


def _ema(values: List[float], period: int) -> float:
    if not values:
        return 0.0
    k = 2 / (period + 1)
    out = float(values[0])
    for v in values[1:]:
        out = float(v) * k + out * (1 - k)
    return out


def _median(vals: List[float]) -> float:
    vals = sorted([float(v) for v in vals if v is not None])
    if not vals:
        return 0.0
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def _build_historical_symbols(
    all_candles: Dict[str, list],
    as_of_ms: int,
    lookback_days: int,
    top_n: int,
    core_symbols: Iterable[str] = ("BTCUSDT", "ETHUSDT"),
) -> Tuple[List[str], List[dict]]:
    """
    Canlı haftalık sembol rotasyonunu geçmişte taklit eder.
    Sadece as_of_ms ÖNCESİ mumları kullanır; gelecek veri kullanmaz.
    Skor: likidite + 30g momentum + win_days + EMA5>EMA20.
    """
    lookback_ms = int(max(7, lookback_days) * 86400 * 1000)
    start_ms = as_of_ms - lookback_ms
    rows = []
    for sym, candles in all_candles.items():
        hist = [c for c in candles if start_ms <= int(c.get("open_time", 0)) < as_of_ms]
        if len(hist) < 50:
            continue
        closes = [float(c["close"]) for c in hist]
        quote_vols = [float(c["close"]) * float(c.get("volume", 0.0)) for c in hist]
        first = closes[0]
        last = closes[-1]
        change_pct = (last - first) / first * 100 if first > 0 else 0.0
        day_changes = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            if prev > 0:
                day_changes.append((closes[i] - prev) / prev * 100)
        win_days = sum(1 for x in day_changes if x > 0) / len(day_changes) if day_changes else 0.0
        ema5 = _ema(closes[-20:], 5)
        ema20 = _ema(closes[-40:], 20)
        ema_ok = ema5 > ema20
        med_qv = _median(quote_vols[-min(len(quote_vols), 24 * 7):])
        avg_qv = sum(quote_vols[-min(len(quote_vols), 24 * 7):]) / max(1, min(len(quote_vols), 24 * 7))

        # Normalize edilmiş, kaba ama geleceksiz skor.
        vol_score = min(1.0, med_qv / 5_000_000.0)
        mom_score = max(0.0, min(1.0, (change_pct + 20.0) / 50.0))
        win_score = max(0.0, min(1.0, win_days))
        ema_score = 1.0 if ema_ok else 0.0
        core_bonus = 0.08 if sym in set(core_symbols) else 0.0
        score = 0.45 * vol_score + 0.25 * mom_score + 0.20 * win_score + 0.10 * ema_score + core_bonus
        rows.append({
            "symbol": sym,
            "score": round(score, 6),
            "change_pct": round(change_pct, 3),
            "win_days": round(win_days, 3),
            "ema_ok": ema_ok,
            "median_quote_volume": round(med_qv, 2),
            "avg_quote_volume": round(avg_qv, 2),
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    selected = [r["symbol"] for r in rows[:top_n]]
    # Core semboller aday havuzunda varsa ve veri yeterliyse listenin en altından yer açarak koru.
    for core in core_symbols:
        if core in all_candles and core not in selected:
            core_row = next((r for r in rows if r["symbol"] == core), None)
            if core_row:
                if len(selected) >= top_n and selected:
                    selected[-1] = core
                else:
                    selected.append(core)
    # Sıra tekrar normalize.
    selected = list(dict.fromkeys(selected))[:top_n]
    return selected, [r for r in rows if r["symbol"] in set(selected)]


def _build_weekly_universe_schedule(
    all_candles: Dict[str, list],
    start_date: str,
    end_date: str,
    lookback_days: int,
    refresh_days: int,
    top_n: int,
) -> List[dict]:
    schedule = []
    for ws, we in _week_ranges(start_date, end_date, refresh_days):
        s_ms = _to_ms_utc(_parse_date(ws))
        e_ms = _to_ms_utc(_parse_date(we))
        symbols, detail = _build_historical_symbols(all_candles, s_ms, lookback_days, top_n)
        schedule.append({"start": ws, "end": we, "start_ms": s_ms, "end_ms": e_ms, "symbols": symbols, "detail": detail})
    return schedule


def _universe_for_ts(schedule: List[dict], ts_ms: int) -> set:
    for row in schedule:
        if row["start_ms"] <= ts_ms < row["end_ms"]:
            return set(row["symbols"])
    return set()


def _fetch_all(symbols: List[str], interval: str, start_ms: int, end_ms: int) -> Dict[str, list]:
    """Tüm sembollerin verisini çeker/cache'ler."""
    all_candles = {}
    start_s = _date_str(datetime.utcfromtimestamp(start_ms / 1000))
    end_s = _date_str(datetime.utcfromtimestamp(end_ms / 1000))
    total_days = max(1, (end_ms - start_ms) // (86400 * 1000))

    for i, sym in enumerate(symbols, 1):
        cached = _load_cache(sym, interval, total_days, start_s, end_s)
        if cached is not None:
            all_candles[sym] = cached
            print(f"  [{i:2}/{len(symbols)}] {sym:<14} cache ({len(cached)} mum)")
            continue

        print(f"  [{i:2}/{len(symbols)}] {sym:<14} indiriliyor...", end=" ", flush=True)
        candles = fetch_klines(sym, interval, start_ms, end_ms)
        if candles:
            _save_cache(sym, interval, total_days, candles, start_s, end_s)
            all_candles[sym] = candles
            print(f"{len(candles)} mum")
        else:
            print("veri yok")
    return all_candles


def _slice_candles(all_candles: Dict[str, list], start_ms: int, end_ms: int) -> Dict[str, list]:
    """Tüm veriden [start_ms, end_ms) aralığını keser."""
    out = {}
    for sym, candles in all_candles.items():
        seg = [c for c in candles if start_ms <= int(c["open_time"]) < end_ms]
        if seg:
            out[sym] = seg
    return out


def _candles_before(all_candles: Dict[str, list], start_ms: int, lookback_limit: int = 500) -> Dict[str, list]:
    """Segment başlangıcından önceki son N mumu warmup için döndürür."""
    out = {}
    for sym, candles in all_candles.items():
        prev = [c for c in candles if int(c["open_time"]) < start_ms]
        if prev:
            out[sym] = prev[-lookback_limit:]
    return out


def _safe_num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _monthly_metrics(trades: list, equity_curve: list, starting_equity: float) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "n": 0, "wr": 0.0, "pnl": 0.0, "dd": 0.0, "sharpe": 0.0,
            "pf": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "rr": 0.0,
            "recovery": 0.0, "max_consec_loss": 0, "best_symbol": "", "worst_symbol": "",
        }

    wins = [t for t in trades if _safe_num(t.get("net_pnl")) > 0]
    losses = [t for t in trades if _safe_num(t.get("net_pnl")) <= 0]
    gross_profit = sum(_safe_num(t.get("net_pnl")) for t in wins)
    gross_loss = abs(sum(_safe_num(t.get("net_pnl")) for t in losses))
    pnl = gross_profit - gross_loss
    wr = len(wins) / n * 100
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = -gross_loss / len(losses) if losses else 0.0
    rr = abs(avg_win / avg_loss) if avg_loss else 0.0
    dd = _max_drawdown(equity_curve)
    sh = _sharpe(equity_curve)
    recovery = pnl / (dd / 100 * starting_equity) if dd > 0 else (999.0 if pnl > 0 else 0.0)

    cur_l = max_l = 0
    for t in trades:
        if _safe_num(t.get("net_pnl")) <= 0:
            cur_l += 1
            max_l = max(max_l, cur_l)
        else:
            cur_l = 0

    by_symbol = defaultdict(float)
    for t in trades:
        by_symbol[str(t.get("symbol", ""))] += _safe_num(t.get("net_pnl"))
    best_symbol = max(by_symbol, key=by_symbol.get) if by_symbol else ""
    worst_symbol = min(by_symbol, key=by_symbol.get) if by_symbol else ""

    return {
        "n": n, "wr": wr, "pnl": pnl, "dd": dd, "sharpe": sh, "pf": pf,
        "avg_win": avg_win, "avg_loss": avg_loss, "rr": rr,
        "recovery": recovery, "max_consec_loss": max_l,
        "best_symbol": best_symbol, "best_symbol_pnl": by_symbol.get(best_symbol, 0.0),
        "worst_symbol": worst_symbol, "worst_symbol_pnl": by_symbol.get(worst_symbol, 0.0),
    }


def _reason_counts(trades: list) -> dict:
    out = defaultdict(int)
    for t in trades:
        out[str(t.get("reason", ""))] += 1
    return dict(out)


def _preload_warmup_buffers(bt: Backtester, seg_symbols: Iterable[str], warmup_candles: Dict[str, list], window: int = 500):
    """LTF buffer'larını ay başından önceki mumlarla doldurur. İşlem açmaz."""
    price_buf = {s: deque(maxlen=window) for s in seg_symbols}
    high_buf = {s: deque(maxlen=window) for s in seg_symbols}
    low_buf = {s: deque(maxlen=window) for s in seg_symbols}
    vol_buf = {s: deque(maxlen=window) for s in seg_symbols}

    for sym in seg_symbols:
        for c in warmup_candles.get(sym, [])[-window:]:
            price_buf[sym].append(float(c["close"]))
            high_buf[sym].append(float(c["high"]))
            low_buf[sym].append(float(c["low"]))
            vol_buf[sym].append(float(c["volume"]))

    # BTC filtresi için geçmiş BTC kapanışlarını da ısıt.
    if "BTCUSDT" in warmup_candles:
        for c in warmup_candles["BTCUSDT"][-window:]:
            bt.btc_closes.append(float(c["close"]))

    return price_buf, high_buf, low_buf, vol_buf


def _preload_htf_buffers(bt: Backtester, seg_symbols: Iterable[str], all_htf_candles: Dict[str, list], seg_start_ms: int, window: int = 500):
    """HTF buffer'larını ay başlangıcından önceki HTF mumlarıyla doldurur."""
    htf_timeline = {}
    htf_ptr = {}

    for sym in seg_symbols:
        bt.htf_prices[sym] = deque(maxlen=window)
        bt.htf_highs[sym] = deque(maxlen=window)
        bt.htf_lows[sym] = deque(maxlen=window)
        bt.htf_volumes[sym] = deque(maxlen=window)

        htf_list = [(int(c["open_time"]), c) for c in all_htf_candles.get(sym, [])]
        htf_list.sort(key=lambda x: x[0])

        ptr = 0
        for ts, c in htf_list:
            if ts < seg_start_ms:
                bt.htf_prices[sym].append(float(c["close"]))
                bt.htf_highs[sym].append(float(c["high"]))
                bt.htf_lows[sym].append(float(c["low"]))
                bt.htf_volumes[sym].append(float(c["volume"]))
                ptr += 1
            else:
                break

        htf_timeline[sym] = htf_list
        htf_ptr[sym] = ptr

    return htf_timeline, htf_ptr


def run_walk_forward(
    symbols: List[str],
    interval: str,
    start_date: str,
    end_date: str,
    cfg: dict,
    out_dir: str | None = None,
    warmup_days: int = 30,
    universe_mode: str = "fixed_current_symbols",
    weekly_symbols: bool = False,
    symbol_lookback_days: int = 30,
    symbol_refresh_days: int = 7,
    candidate_symbols: List[str] | None = None,
):
    months = _month_ranges(start_date, end_date)

    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    fetch_start_dt = start_dt - timedelta(days=max(0, warmup_days))

    start_ms = _to_ms_utc(start_dt)
    end_ms = _to_ms_utc(end_dt)
    fetch_start_ms = _to_ms_utc(fetch_start_dt)

    sep = "=" * 100
    print(f"\n{sep}")
    print(f"  AYLIK ROBUSTLUK ANALİZİ / WF-BENZERİ TEST  —  {len(months)} ay")
    print(f"  Test: {start_date} → {end_date}  |  Veri başlangıcı warmup: {_date_str(fetch_start_dt)}")
    print(f"  Sembol: {len(symbols)}  |  Interval: {interval}  |  Warmup: {warmup_days} gün")
    print(f"  Universe mode: {universe_mode}")
    print(sep)

    if universe_mode == "fixed_current_symbols":
        print("\n[UYARI] symbols_top70.json bugünün listesi ise geçmiş aylarda look-ahead bias oluşturabilir.")
        print("        Daha temiz test için --symbols ile sabit majör/likit liste ver veya tarihsel universe üretici ekle.")

    print("\n  LTF veri yükleniyor...")
    # Haftalık sembol rotasyonu aktifse, sadece ilk TOP sembolü değil,
    # aday havuzunu da geçmiş veriyle yüklemek gerekir. Aksi halde haftalık
    # evren bugünün ilk TOP listesine sıkışır ve rotasyon gerçek çalışmaz.
    fetch_symbols = list(candidate_symbols or symbols) if weekly_symbols else list(symbols)
    fetch_symbols = list(dict.fromkeys([s.upper() for s in fetch_symbols]))
    all_candles = _fetch_all(fetch_symbols, interval, fetch_start_ms, end_ms)
    if not all_candles:
        print("\n[HATA] Veri yüklenemedi.")
        return []

    # BTCUSDT, sembol listesinde yoksa bile BTC filtresi için veri gerekebilir.
    if cfg.get("btc_filter", {}).get("enabled", False) and "BTCUSDT" not in all_candles:
        print("  BTCUSDT BTC filtresi için ekleniyor...")
        btc_data = _fetch_all(["BTCUSDT"], interval, fetch_start_ms, end_ms)
        all_candles.update(btc_data)

    # Haftalık tarihsel sembol evreni oluştur.
    # Sadece her haftanın başlangıcından ÖNCEKİ lookback verisi kullanılır.
    # Bu sayede canlıdaki haftalık sembol rotasyonuna benzer davranır ve
    # gelecekteki mumlarla sembol seçimi yapılmaz.
    weekly_schedule = []
    if weekly_symbols:
        universe_mode = "weekly_historical_symbols"
        print(f"\n  Haftalık sembol evreni oluşturuluyor: lookback={symbol_lookback_days}g, refresh={symbol_refresh_days}g, top={len(symbols)}")
        weekly_schedule = _build_weekly_universe_schedule(
            all_candles=all_candles,
            start_date=start_date,
            end_date=end_date,
            lookback_days=symbol_lookback_days,
            refresh_days=symbol_refresh_days,
            top_n=len(symbols),
        )
        if not weekly_schedule:
            print("\n[HATA] Haftalık sembol evreni oluşturulamadı.")
            return []
        for row in weekly_schedule[:3]:
            print(f"    {row['start']} → {row['end']}: {', '.join(row['symbols'][:8])}{'...' if len(row['symbols']) > 8 else ''}")
        if len(weekly_schedule) > 3:
            print(f"    ... toplam {len(weekly_schedule)} haftalık evren")

    # Funding rate yükle.
    fr_cfg = cfg.get("funding_filter", {})
    funding_map = {}
    if fr_cfg.get("enabled", False):
        print("  BTC funding rate yükleniyor...")
        funding_map = fetch_funding_rates("BTCUSDT", fetch_start_ms, end_ms)
        print(f"  {len(funding_map)} funding kaydı yüklendi")

    # HTF verisi yükle.
    htf_cfg = cfg.get("mtf", {})
    htf_enabled = bool(htf_cfg.get("enabled", False))
    htf_interval = htf_cfg.get("htf_interval", "1h")
    all_htf_candles = {}
    if htf_enabled:
        print(f"  HTF veri yükleniyor ({htf_interval})...")
        for sym in list(all_candles.keys()):
            cached = _load_cache(sym, htf_interval, max(1, warmup_days), _date_str(fetch_start_dt), end_date)
            if cached is not None:
                all_htf_candles[sym] = cached
                continue
            candles = fetch_klines(sym, htf_interval, fetch_start_ms, end_ms)
            if candles:
                _save_cache(sym, htf_interval, max(1, warmup_days), candles, _date_str(fetch_start_dt), end_date)
                all_htf_candles[sym] = candles
        print(f"  HTF yüklendi: {len(all_htf_candles)} sembol")

    results = []
    print(f"\n{sep}")
    print(f"  {'Ay':<10} {'İşlem':>6} {'WR':>6} {'PnL':>10} {'PF':>7} {'DD':>7} {'Rec':>7} {'MaxL':>5} {'Best':>12} {'Worst':>12}")
    print("-" * 100)

    for seg_start, seg_end in months:
        s_ms = _to_ms_utc(_parse_date(seg_start))
        e_ms = _to_ms_utc(_parse_date(seg_end))
        month_key = seg_start[:7]

        seg_candles = _slice_candles(all_candles, s_ms, e_ms)
        if weekly_symbols:
            month_weeks = [w for w in weekly_schedule if not (w["end_ms"] <= s_ms or w["start_ms"] >= e_ms)]
            month_union = []
            for w in month_weeks:
                for sym in w["symbols"]:
                    if sym not in month_union:
                        month_union.append(sym)
            seg_symbols = [s for s in month_union if s in seg_candles]
        else:
            # BTC sadece filtre için eklendiyse ve kullanıcı universe'inde yoksa trade universe'ten çıkar.
            seg_symbols = [s for s in symbols if s in seg_candles]
        if not seg_symbols:
            results.append({"month": month_key, "n": 0, "wr": 0, "pnl": 0, "dd": 0, "sharpe": 0, "pf": 0})
            print(f"  {month_key:<10} {'0':>6} {'—':>6} {'$0':>10} {'—':>7} {'—':>7} {'—':>7} {'—':>5} {'—':>12} {'—':>12}")
            continue

        bt = Backtester(cfg)
        bt._funding_map = {k: v for k, v in funding_map.items() if s_ms <= int(k) < e_ms}

        warmup_candles = _candles_before(all_candles, s_ms, lookback_limit=500)
        price_buf, high_buf, low_buf, vol_buf = _preload_warmup_buffers(bt, seg_symbols, warmup_candles, window=500)

        if htf_enabled:
            htf_timeline, htf_ptr = _preload_htf_buffers(bt, seg_symbols, all_htf_candles, s_ms, window=500)
        else:
            htf_timeline, htf_ptr = {}, {}

        timeline = []
        for sym in seg_symbols:
            for c in seg_candles.get(sym, []):
                timeline.append((int(c["open_time"]), sym, c))
        timeline.sort(key=lambda x: x[0])

        last_prices = {}
        for ts, sym, candle in timeline:
            # HTF: sadece mevcut LTF mum zamanına kadar oluşmuş HTF mumlarını ekle.
            if htf_enabled and sym in htf_timeline:
                ptr = htf_ptr.get(sym, 0)
                htf_list = htf_timeline[sym]
                while ptr < len(htf_list) and htf_list[ptr][0] <= ts:
                    hc = htf_list[ptr][1]
                    bt.htf_prices[sym].append(float(hc["close"]))
                    bt.htf_highs[sym].append(float(hc["high"]))
                    bt.htf_lows[sym].append(float(hc["low"]))
                    bt.htf_volumes[sym].append(float(hc["volume"]))
                    ptr += 1
                htf_ptr[sym] = ptr

            price_buf[sym].append(float(candle["close"]))
            high_buf[sym].append(float(candle["high"]))
            low_buf[sym].append(float(candle["low"]))
            vol_buf[sym].append(float(candle["volume"]))
            last_prices[sym] = float(candle["close"])

            if len(price_buf[sym]) >= 50:
                if weekly_symbols:
                    current_universe = _universe_for_ts(weekly_schedule, ts)
                    # Canlıya yakın davranış: yeni pozisyon sadece o haftanın evrenindeki sembolde açılır.
                    # Ancak sembol evrenden düşse bile açık pozisyon varsa mum beslenir ve pozisyon yönetilir.
                    if sym not in current_universe and sym not in bt.open_positions:
                        continue
                bt.step(sym, candle, list(price_buf[sym]), list(high_buf[sym]), list(low_buf[sym]), list(vol_buf[sym]))

        last_ts_ms = timeline[-1][0] if timeline else e_ms - 1
        bt.force_close_all(last_prices, last_ts_ms=last_ts_ms)

        m = _monthly_metrics(bt.trades, bt.equity_curve, bt.starting_equity)
        m.update({"month": month_key, "exit_reasons": _reason_counts(bt.trades)})
        results.append(m)

        flag = " ⚠️" if m["pnl"] < -100 else ("" if m["pnl"] >= 0 else " ⚡")
        best = f"{m.get('best_symbol','')[:7]} {m.get('best_symbol_pnl',0):+.0f}" if m.get("best_symbol") else "—"
        worst = f"{m.get('worst_symbol','')[:7]} {m.get('worst_symbol_pnl',0):+.0f}" if m.get("worst_symbol") else "—"
        print(f"  {month_key:<10} {m['n']:>6} {m['wr']:>5.0f}% ${m['pnl']:>+9.0f} {m['pf']:>7.2f} {m['dd']:>6.1f}% {m['recovery']:>7.2f} {m['max_consec_loss']:>5} {best:>12} {worst:>12}{flag}")

        if out_dir:
            month_dir = Path(out_dir) / month_key
            generate_report(
                trades=bt.trades,
                starting_equity=bt.starting_equity,
                final_equity=bt.equity,
                equity_curve=bt.equity_curve,
                out_dir=month_dir,
                label=month_key,
                sl_records=bt.sl_records,
                all_candles={s: seg_candles[s] for s in seg_symbols if s in seg_candles},
                block_log=bt.block_log,
                tp_records=bt.tp_records,
            )

    active = [r for r in results if r.get("n", 0) > 0]
    if not active:
        print("\n[UYARI] Hiç işlem oluşmadı.")
        return results

    pnls = [r["pnl"] for r in active]
    total = sum(pnls)
    avg = total / len(active)
    pos_m = sum(1 for p in pnls if p > 0)
    neg_m = sum(1 for p in pnls if p < 0)
    worst_row = min(active, key=lambda r: r["pnl"])
    best_row = max(active, key=lambda r: r["pnl"])
    worst = worst_row["pnl"]
    best = best_row["pnl"]
    var = sum((p - avg) ** 2 for p in pnls) / len(pnls)
    std = var ** 0.5
    total_gross_profit = sum(max(0.0, r["pnl"]) for r in active)
    total_gross_loss = abs(sum(min(0.0, r["pnl"]) for r in active))
    month_pf = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 999.0
    avg_dd = sum(r["dd"] for r in active) / len(active)
    worst_dd = max(r["dd"] for r in active)
    avg_pf = sum(r["pf"] for r in active if r["pf"] < 999) / max(1, sum(1 for r in active if r["pf"] < 999))
    consistency_score = avg / std if std > 0 else 0.0

    print(f"\n{sep}")
    print("  ROBUSTLUK ÖZETİ")
    print(sep)
    print(f"  Toplam PnL             : ${total:+.2f}")
    print(f"  Aylık ortalama         : ${avg:+.2f}")
    print(f"  Pozitif ay             : {pos_m}/{len(active)}")
    print(f"  Negatif ay             : {neg_m}/{len(active)}")
    print(f"  En kötü ay             : ${worst:+.2f} ({worst_row['month']})")
    print(f"  En iyi ay              : ${best:+.2f} ({best_row['month']})")
    print(f"  Aylık std sapma        : ${std:.2f}")
    print(f"  Tutarlılık skoru       : {consistency_score:.3f}")
    print(f"  Aylık PF               : {month_pf:.2f}")
    print(f"  Ortalama trade PF      : {avg_pf:.2f}")
    print(f"  Ortalama DD            : %{avg_dd:.2f}")
    print(f"  En kötü DD             : %{worst_dd:.2f}")

    print("\n  DEĞERLENDİRME:")
    if pos_m / len(active) < 0.60:
        print("  ✗ Pozitif ay oranı düşük — strateji rejime fazla bağımlı olabilir.")
    elif worst < -200:
        print("  ⚠ En kötü ay çok sert — kötü dönem tek başına birikimi silebilir.")
    elif month_pf < 1.25:
        print("  ⚠ Aylık PF zayıf — pozitif görünüm birkaç aya bağımlı olabilir.")
    elif worst_dd > 15:
        print("  ⚠ En kötü DD yüksek — canlı risk azaltılmalı.")
    else:
        print("  ✓ Aylık dağılım kabul edilebilir — gerçek WF/uzun OOS testine geçilebilir.")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "wf_monthly.csv")
        fields = [
            "month", "n", "wr", "pnl", "pf", "dd", "sharpe", "avg_win", "avg_loss", "rr",
            "recovery", "max_consec_loss", "best_symbol", "best_symbol_pnl", "worst_symbol", "worst_symbol_pnl",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter=";")
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"  Aylık özet             → {csv_path}")

        # Exit reason özetini ayrı yaz.
        reason_path = os.path.join(out_dir, "wf_exit_reasons_by_month.csv")
        all_reasons = sorted({reason for r in results for reason in r.get("exit_reasons", {}).keys()})
        with open(reason_path, "w", newline="", encoding="utf-8-sig") as f:
            fields2 = ["month"] + all_reasons
            w = csv.DictWriter(f, fieldnames=fields2, delimiter=";")
            w.writeheader()
            for r in results:
                row = {"month": r.get("month", "")}
                for reason in all_reasons:
                    row[reason] = r.get("exit_reasons", {}).get(reason, 0)
                w.writerow(row)
        print(f"  Exit reason özeti      → {reason_path}")

        if weekly_symbols:
            uni_path = os.path.join(out_dir, "wf_symbol_universe_by_week.csv")
            detail_path = os.path.join(out_dir, "wf_symbol_universe_details_by_week.csv")
            with open(uni_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=["week_start", "week_end", "symbols"], delimiter=";")
                w.writeheader()
                for row in weekly_schedule:
                    w.writerow({"week_start": row["start"], "week_end": row["end"], "symbols": ",".join(row["symbols"])})
            with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
                fields_u = ["week_start", "week_end", "symbol", "rank", "score", "change_pct", "win_days", "ema_ok", "median_quote_volume"]
                w = csv.DictWriter(f, fieldnames=fields_u, delimiter=";")
                w.writeheader()
                for row in weekly_schedule:
                    rank_map = {s: i + 1 for i, s in enumerate(row["symbols"])}
                    for d in row["detail"]:
                        if d["symbol"] in rank_map:
                            w.writerow({
                                "week_start": row["start"], "week_end": row["end"],
                                "symbol": d["symbol"], "rank": rank_map[d["symbol"]],
                                "score": d.get("score", ""), "change_pct": d.get("change_pct", ""),
                                "win_days": d.get("win_days", ""), "ema_ok": d.get("ema_ok", ""),
                                "median_quote_volume": d.get("median_quote_volume", ""),
                            })
            print(f"  Haftalık evren         → {uni_path}")
            print(f"  Evren detayları        → {detail_path}")

        wf_summary = {
            "test_type": "monthly_robustness_not_true_train_optimize_test_wf",
            "start_date": start_date,
            "end_date": end_date,
            "fetch_start_date": _date_str(fetch_start_dt),
            "interval": interval,
            "symbols": len(symbols),
            "candidate_symbols": len(candidate_symbols or symbols),
            "universe_mode": universe_mode,
            "weekly_symbols": bool(weekly_symbols),
            "symbol_lookback_days": symbol_lookback_days,
            "symbol_refresh_days": symbol_refresh_days,
            "warmup_days": warmup_days,
            "total_pnl": round(total, 2),
            "avg_monthly": round(avg, 2),
            "pos_months": pos_m,
            "neg_months": neg_m,
            "best_pnl": round(best, 2),
            "best_month": best_row["month"],
            "worst_pnl": round(worst, 2),
            "worst_month": worst_row["month"],
            "std_dev": round(std, 2),
            "consistency_score": round(consistency_score, 3),
            "monthly_pf": round(month_pf, 3),
            "avg_trade_pf": round(avg_pf, 3),
            "avg_dd": round(avg_dd, 2),
            "worst_dd": round(worst_dd, 2),
            "verdict": "positive" if (pos_m / len(active) >= 0.60 and worst >= -200 and month_pf >= 1.25 and worst_dd <= 15) else "caution",
        }
        json_path = os.path.join(out_dir, "wf_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(wf_summary, f, ensure_ascii=False, indent=2)
        print(f"  Genel özet             → {json_path}")
        print(f"  Aylık detay raporları  → {out_dir}/<ay>/backtest_*.csv")

    return results


def _load_symbols(args) -> tuple[List[str], str, List[str]]:
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        return syms[:args.top], "manual_symbols", syms

    sym_path = Path(args.symbols_file) if args.symbols_file else Path(_SCRIPT_DIR) / "symbols_top70.json"
    all_symbols = [s.upper() for s in json.loads(sym_path.read_text(encoding="utf-8"))]
    candidate_top = int(getattr(args, "candidate_top", 0) or 0)
    candidates = all_symbols[:candidate_top] if candidate_top > 0 else all_symbols
    return all_symbols[:args.top], "fixed_current_symbols", candidates


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Aylık Robustluk / WF-benzeri Analiz")
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD, exclusive")
    p.add_argument("--interval", type=str, default="1h")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--symbols", type=str, default="", help="Virgüllü manuel sembol listesi")
    p.add_argument("--symbols-file", type=str, default="", help="Alternatif symbols json dosyası")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--warmup-days", type=int, default=30)
    p.add_argument("--weekly-symbols", action="store_true", help="Canlıya yakın: her 7 günde bir sadece geçmiş lookback ile sembol evreni seç")
    p.add_argument("--symbol-lookback-days", type=int, default=30, help="Haftalık sembol seçimi için geçmiş gün sayısı")
    p.add_argument("--symbol-refresh-days", type=int, default=7, help="Sembol evreni kaç günde bir yenilensin")
    p.add_argument("--candidate-top", type=int, default=0, help="symbols_file içinden kaç aday sembol yüklensin; 0=tümü")
    args = p.parse_args()

    cfg_path = _os.path.join(_SCRIPT_DIR, "config_online.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    symbols, universe_mode, candidate_symbols = _load_symbols(args)
    run_walk_forward(
        symbols=symbols,
        interval=args.interval,
        start_date=args.start,
        end_date=args.end,
        cfg=cfg,
        out_dir=args.out or None,
        warmup_days=args.warmup_days,
        universe_mode=universe_mode,
        weekly_symbols=args.weekly_symbols,
        symbol_lookback_days=args.symbol_lookback_days,
        symbol_refresh_days=args.symbol_refresh_days,
        candidate_symbols=candidate_symbols,
    )
