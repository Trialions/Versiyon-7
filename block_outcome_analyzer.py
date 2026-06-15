# block_outcome_analyzer.py — Engel Yiyen Sinyallerin Sonuç Analizi
#
# Mevcut ghost_signal_analysis.csv üzerine şunları ekler:
#  1. first_hit  : TP mi önce geldi, SL mi?  (en kritik metrik)
#  2. 8h pencere : 4/8/12/24 tam dörtlü pencere
#  3. cooldown   : aynı sembol+sebep için tekrar kaydı engeller
#  4. by_regime  : rejime göre özet rapor
#  5. verdict    : GEREKSIZ_ENGEL / DOGRU_ENGEL / BELIRSIZ / KACAN_TREND
#
# Kullanım (generate_report içinden):
#   from block_outcome_analyzer import build_block_outcome, write_block_outcome_reports

from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ── Yardımcı: ms timestamp parse ──────────────────────────────
def _parse_ms(time_str: str) -> Optional[int]:
    """'2025-01-15 14:35' veya '2025-01-15 14:35:00' → ms"""
    if not time_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            import calendar, datetime
            dt = datetime.datetime.strptime(time_str.strip(), fmt)
            return int(calendar.timegm(dt.timetuple()) * 1000)
        except ValueError:
            continue
    return None


# ── Yardımcı: mum listesinde ts_ms'e en yakın mumu bul ────────
def _candle_idx_at(candles: list, ts_ms: int) -> int:
    """ts_ms'e >= olan ilk mum index'ini döner."""
    lo, hi = 0, len(candles) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if int(candles[mid].get("open_time", 0)) < ts_ms:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ── first_hit hesabı ─────────────────────────────────────────
def _first_hit(candles: list, start_idx: int, entry: float,
               tp_pct: float, sl_pct: float,
               horizon_ms: int, start_ms: int) -> str:
    """
    start_idx'ten itibaren horizon_ms süre içinde TP mi SL mi önce geldi?
    tp_pct / sl_pct decimal (0.04 = %4)
    Dönüş: "TP_FIRST" | "SL_FIRST" | "BOTH" | "NONE"
    """
    tp_price = entry * (1 + tp_pct)
    sl_price = entry * (1 - sl_pct)
    end_ms   = start_ms + horizon_ms

    tp_bar = sl_bar = None

    for i, c in enumerate(candles[start_idx:], start=start_idx):
        if int(c.get("open_time", 0)) > end_ms:
            break
        high = float(c.get("high", entry))
        low  = float(c.get("low",  entry))
        if tp_bar is None and high >= tp_price:
            tp_bar = i
        if sl_bar is None and low  <= sl_price:
            sl_bar = i

    if tp_bar is None and sl_bar is None:
        return "NONE"
    if tp_bar is not None and sl_bar is None:
        return "TP_FIRST"
    if sl_bar is not None and tp_bar is None:
        return "SL_FIRST"
    if tp_bar == sl_bar:
        return "BOTH"  # Aynı mumda hem TP hem SL — sıra bilinmiyor
    return "TP_FIRST" if tp_bar < sl_bar else "SL_FIRST"


# ── Pencere metrikleri ────────────────────────────────────────
def _window_metrics(candles: list, start_idx: int, entry: float,
                    tp_pct: float, sl_pct: float,
                    horizon_hours: int, start_ms: int) -> dict:
    """Bir pencere için tüm metrikleri hesaplar."""
    horizon_ms = horizon_hours * 3600 * 1000
    end_ms     = start_ms + horizon_ms

    future = [c for c in candles[start_idx:]
               if int(c.get("open_time", 0)) <= end_ms]

    if not future:
        return {
            "max_up_pct": None, "max_down_pct": None,
            "close_return_pct": None,
            "first_hit": "NONE", "verdict": "NO_DATA",
        }

    max_up   = max((float(c.get("high",  entry)) - entry) / entry * 100 for c in future)
    max_down = min((float(c.get("low",   entry)) - entry) / entry * 100 for c in future)
    close_r  = (float(future[-1].get("close", entry)) - entry) / entry * 100

    first = _first_hit(candles, start_idx, entry,
                       tp_pct, sl_pct, horizon_ms, start_ms)

    # Verdict
    tp_threshold = tp_pct * 100   # decimal → yüzde
    sl_threshold = sl_pct * 100

    if first == "TP_FIRST":
        verdict = "GEREKSIZ_ENGEL"
    elif first == "SL_FIRST":
        verdict = "DOGRU_ENGEL"
    elif first == "BOTH":
        verdict = "BELIRSIZ_VOLATIL"  # Aynı mumda TP+SL, sıra bilinmiyor
    elif max_up >= tp_threshold and abs(max_down) < sl_threshold * 0.5:
        verdict = "KACAN_TREND"
    elif max_up >= tp_threshold and abs(max_down) >= sl_threshold:
        verdict = "BELIRSIZ_VOLATIL"
    elif abs(max_down) >= sl_threshold:
        verdict = "DOGRU_ENGEL"
    else:
        verdict = "BELIRSIZ"

    return {
        "max_up_pct":       round(max_up,   3),
        "max_down_pct":     round(max_down,  3),
        "close_return_pct": round(close_r,   3),
        "first_hit":        first,
        "verdict":          verdict,
    }


# ── Ana analiz fonksiyonu ─────────────────────────────────────
def build_block_outcome(
    block_log: list,
    all_candles: dict,
    tp_pct:          float = 0.04,   # decimal (0.04 = %4)
    sl_pct:          float = 0.03,   # decimal (0.03 = %3)
    horizons_hours:  list  = None,   # [4, 8, 12, 24]
    cooldown_bars:   int   = 12,     # 5m için = 1 saat
    bar_seconds:     int   = 300,    # 5m = 300s
    only_reasons:    set   = None,
    max_per_reason:  int   = 5000,
) -> list:
    """
    block_log içindeki engel kayıtlarını analiz eder.
    Her sembol+sebep için cooldown uygulanır.
    Her kayıt için 4 pencerede (4/8/12/24h) metrikler hesaplanır.
    """
    if horizons_hours is None:
        horizons_hours = [4, 8, 12, 24]

    if only_reasons is None:
        only_reasons = {
            "SCORE_THRESHOLD", "REGIME_CLOSED", "LOW_VOLUME",
            "ADX_FILTER", "LOW_NOTIONAL", "MTF_NO_CONFIRM",
            "RSI_TOO_HIGH", "RSI_TOO_LOW", "ATR_TOO_LOW",
            "SYMBOL_BLACKLIST", "WEAK_SYMBOL_LOW_QUALITY",
            "KONSOL_LOW_SCORE", "KONSOL_WEAK_ADX",
            "KONSOL_LOW_ATR", "KONSOL_NO_VOLUME_BREAKOUT",
            "KONSOL_HTF_WEAK",
        }

    cooldown_ms        = cooldown_bars * bar_seconds * 1000
    last_seen: dict    = {}   # (symbol, reason) → last_ts_ms
    reason_count: dict = defaultdict(int)
    rows               = []

    for r in block_log:
        sym    = r.get("symbol", "")
        reason = r.get("cause",  r.get("reason", ""))

        if not sym or sym not in all_candles:
            continue
        if only_reasons and reason not in only_reasons:
            continue
        if reason_count[reason] >= max_per_reason:
            continue

        ts_ms = _parse_ms(r.get("time", ""))
        if ts_ms is None:
            continue

        # Cooldown kontrolü
        key      = (sym, reason)
        last_ts  = last_seen.get(key, 0)
        if ts_ms - last_ts < cooldown_ms:
            continue
        last_seen[key] = ts_ms

        candles = all_candles[sym]
        if not candles:
            continue

        start_idx = _candle_idx_at(candles, ts_ms)
        if start_idx >= len(candles):
            continue

        entry = float(candles[start_idx].get("close", 0.0))
        if entry <= 0:
            continue

        # DÜZ-3: same-candle look-ahead önleme
        # Entry fiyatı mevcut mum kapanışından alınır,
        # outcome analizi bir SONRAKİ mumdan başlar.
        eval_idx = start_idx + 1
        if eval_idx >= len(candles):
            continue

        row = {
            "timestamp":     r.get("time", ""),
            "symbol":        sym,
            "reason":        reason,
            "detail":        r.get("detail", ""),
            "score":         r.get("score",  ""),
            "htf_score":     r.get("htf_score", ""),
            "regime":        r.get("regime", ""),
            "rsi":           r.get("rsi",    ""),
            "adx":           r.get("adx",    ""),
            "atr_pct":       r.get("atr",    ""),
            "volume_ratio":  r.get("vol_ratio", ""),
            "btc_trend_ok":  r.get("btc_trend", ""),
            "entry_price":   round(entry, 6),
        }

        for h in horizons_hours:
            m = _window_metrics(candles, eval_idx, entry,
                                tp_pct, sl_pct, h, ts_ms)
            pfx = f"h{h}_"
            row[pfx + "max_up_pct"]       = m["max_up_pct"]
            row[pfx + "max_down_pct"]     = m["max_down_pct"]
            row[pfx + "close_return_pct"] = m["close_return_pct"]
            row[pfx + "first_hit"]        = m["first_hit"]
            row[pfx + "verdict"]          = m["verdict"]

        rows.append(row)
        reason_count[reason] += 1

    return rows


# ── Özet rapor yazıcısı ───────────────────────────────────────
def _summarize(rows: list, group_key: str,
               horizons: list = None) -> list:
    """rows'u group_key'e göre grupla, her grup için özet üret."""
    if horizons is None:
        horizons = [4, 8, 12, 24]

    groups: dict = defaultdict(list)
    for r in rows:
        groups[r.get(group_key, "UNKNOWN")].append(r)

    summary = []
    for key, grp in sorted(groups.items()):
        n = len(grp)
        rec: dict = {group_key: key, "count": n}

        for h in horizons:
            pfx   = f"h{h}_"
            valid = [r for r in grp if r.get(pfx + "max_up_pct") is not None]
            if not valid:
                continue
            vn = len(valid)

            rec[pfx + "avg_max_up_pct"]       = round(
                sum(r[pfx+"max_up_pct"]       for r in valid) / vn, 3)
            rec[pfx + "avg_max_down_pct"]      = round(
                sum(r[pfx+"max_down_pct"]     for r in valid) / vn, 3)
            rec[pfx + "avg_close_return_pct"]  = round(
                sum(r[pfx+"close_return_pct"] for r in valid) / vn, 3)

            # first_hit dağılımı
            fh = [r[pfx+"first_hit"] for r in valid]
            rec[pfx + "tp_first_rate"]  = round(fh.count("TP_FIRST") / vn * 100, 1)
            rec[pfx + "sl_first_rate"]  = round(fh.count("SL_FIRST") / vn * 100, 1)
            rec[pfx + "none_rate"]      = round(fh.count("NONE")      / vn * 100, 1)

            # verdict dağılımı
            vd = [r[pfx+"verdict"] for r in valid]
            rec[pfx + "gereksiz_pct"]   = round(vd.count("GEREKSIZ_ENGEL")   / vn * 100, 1)
            rec[pfx + "dogru_pct"]      = round(vd.count("DOGRU_ENGEL")      / vn * 100, 1)
            rec[pfx + "kacan_pct"]      = round(vd.count("KACAN_TREND")      / vn * 100, 1)
            rec[pfx + "belirsiz_pct"]   = round(
                (vd.count("BELIRSIZ") + vd.count("BELIRSIZ_VOLATIL")) / vn * 100, 1)

        summary.append(rec)

    # h24 gereksiz_pct'ye göre sırala
    summary.sort(
        key=lambda x: x.get("h24_gereksiz_pct", 0.0),
        reverse=True
    )
    return summary


def write_block_outcome_reports(
    out_dir: Path,
    rows: list,
    horizons: list = None,
):
    """
    Dört CSV raporu yazar:
      block_outcome_analysis.csv        — tüm kayıtlar
      block_outcome_summary_by_reason.csv
      block_outcome_summary_by_symbol.csv
      block_outcome_summary_by_regime.csv
    """
    if horizons is None:
        horizons = [4, 8, 12, 24]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    # 1. Tüm kayıtlar
    detail_path = out_dir / "block_outcome_analysis.csv"
    fieldnames  = list(rows[0].keys())
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # 2-4. Özet raporlar
    for group_key, fname in [
        ("reason",  "block_outcome_summary_by_reason.csv"),
        ("symbol",  "block_outcome_summary_by_symbol.csv"),
        ("regime",  "block_outcome_summary_by_regime.csv"),
    ]:
        summary = _summarize(rows, group_key, horizons)
        if not summary:
            continue
        path = out_dir / fname
        fnames = list(summary[0].keys())
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fnames, delimiter=";",
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(summary)
