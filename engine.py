# engine.py — İşlem motoru v8
# DEĞİŞİKLİKLER v5→v6:
#   1. HTF skor konfirmasyonu eklendi
#   2. htf_close_series / htf_high_series / htf_low_series / htf_vol_series
#   3. seed_from_candles_htf() ve on_candle_htf()
#   4. _try_open'da HTF konfirmasyon şartı
# DEĞİŞİKLİKLER v6→v7 (adaptive_sl):
#   5. adaptive_sl modülü entegre edildi
#   6. _open(): SL mesafesi rejime göre dinamik
#   7. _try_open(): giriş score eşiği rejime göre otomatik ayarlanıyor
#   8. _manage() / _exit_reason(): trail_step pozisyona özel
# DEĞİŞİKLİKLER v7→v8:
#   9.  ÖNCELİK 3: symbol_blacklist — config'den kalıcı sembol kara listesi
#   10. ÖNCELİK 5: vol_position_filter — yüksek ATR'de otomatik pozisyon küçültme
import time
import csv
import threading
from collections import deque
from pathlib import Path

import adaptive_sl
from strategy_core import score_symbol
from data_macro import get_market_sentiment, get_sentiment_score
from market_regime import MarketRegimeDetector
from symbol_manager import SymbolManager
from logger import log_info, log_error, log_event


class TradeEngine:
    def __init__(self, symbols: list, cfg: dict = None, data_dir: str = "data"):
        self.cfg      = cfg or {}
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        risk  = self.cfg.get("risk",        {})
        lim   = self.cfg.get("limits",      {})
        pyr   = self.cfg.get("pyramiding",  {})
        thr   = self.cfg.get("thresholds",  {})
        misc  = self.cfg.get("misc",        {})
        mtf   = self.cfg.get("mtf",         {})
        adx_f = self.cfg.get("adx_filter",  {})
        self.adx_filter_enabled   = bool( adx_f.get("enabled",   True))
        self.adx_filter_threshold = float(adx_f.get("threshold", 25.0))

        # ── ATR Minimum Filtresi ──────────────────────────────
        atr_f = cfg.get("atr_filter", {})
        self.atr_filter_enabled = bool( atr_f.get("enabled",     False))
        self.atr_filter_min     = float(atr_f.get("min_atr_pct", 0.8))

        # ── RSI Filtresi ──────────────────────────────────────
        rsi_f = self.cfg.get("rsi_filter", {})
        self.rsi_filter_enabled = bool( rsi_f.get("enabled",   True))
        self.rsi_max_long       = float(rsi_f.get("max_long",  73.0))
        self.rsi_min_short      = float(rsi_f.get("min_short", 30.0))

        self.equity           = float(misc.get("starting_equity_usdt", 1000.0))
        self.tp_pct           = float(risk.get("take_profit_min_pct",  3.0)) / 100
        self.sl_pct           = float(risk.get("hard_stop_pct",        1.5)) / 100
        self.use_atr_stop     = bool(risk.get("use_atr_stop", True))
        self.atr_multiplier   = float(risk.get("atr_multiplier", 2.0))
        self.max_stop_pct     = float(risk.get("max_stop_pct", 4.5)) / 100
        self.trail            = bool( risk.get("use_trailing",         True))
        self.trail_step       = float(risk.get("trailing_step_pct",    0.7)) / 100
        # Dynamic Trail: kapalıysa adaptive_sl'in ATR bazlı trail_step çıktısı kullanılmaz,
        # klasik risk.trailing_step_pct pozisyona yazılır.
        dt_cfg = self.cfg.get("dynamic_trail", {})
        self.dynamic_trail_enabled = bool(dt_cfg.get("enabled", True))
        self.min_hold         = int(  risk.get("min_hold_minutes",     30))  * 60
        self.risk_per_trade   = float(risk.get("risk_per_trade_pct",   1.0)) / 100
        self.min_profit_close = float(risk.get("min_profit_close_pct", 3.0)) / 100

        self.score_long_open  = float(thr.get("score_long_open",  85))
        self.score_short_open = float(thr.get("score_short_open", 15))
        self.score_close      = float(thr.get("score_close",      50))

        # ── Dynamic Threshold / Rejim Bazlı Giriş Eşiği ───────
        dt = self.cfg.get("dynamic_threshold", {})
        self.dynamic_threshold_enabled = bool(dt.get("enabled", False))
        self.dt_trend_score            = float(dt.get("trend_score", max(80.0, self.score_long_open - 5)))
        self.dt_konsol_score           = float(dt.get("konsol_score", self.score_long_open + 2))
        self.dt_bearish_score          = float(dt.get("bearish_score", 999.0))
        self.dt_neutral_score          = float(dt.get("neutral_score", self.score_long_open))
        self.dt_min_score              = float(dt.get("min_score", 85.0))
        self.dt_max_score              = float(dt.get("max_score", 999.0))
        self.dt_strong_discount        = float(dt.get("strong_setup_discount", 2.0))
        self.dt_strong_htf             = float(dt.get("strong_htf", 75.0))
        self.dt_strong_adx             = float(dt.get("strong_adx", 35.0))
        self.dt_strong_rsi_max         = float(dt.get("strong_rsi_max", 68.0))

        self.max_trades_day   = int(  lim.get("max_trades_per_day", 10))
        self.max_open_pos     = int(  lim.get("max_open_positions",  5))
        self.daily_target_pct = float(lim.get("daily_target_pct",  8.0)) / 100
        self.max_hold_sec     = int(  lim.get("max_hold_hours",       48)) * 3600
        self.daily_loss_limit = float(lim.get("daily_loss_limit_pct",  5.0)) / 100

        self.vol_mult         = float(misc.get("volume_burst_multiplier", 2.0))
        self.min_notional     = float(misc.get("min_notional_usdt", 30000.0))
        self.pyramid_enabled  = bool( pyr.get("enabled",    False))

        # ── Çoklu Timeframe Ayarları ──────────────────────────────
        self.mtf_enabled      = bool( mtf.get("enabled",          True))
        # HTF'de LONG için minimum skor (yön konfirmasyonu)
        self.mtf_long_min     = float(mtf.get("htf_long_min",     55.0))
        # HTF'de SHORT için maksimum skor (yön konfirmasyonu)
        self.mtf_short_max    = float(mtf.get("htf_short_max",    45.0))

        self.lock             = threading.Lock()
        self._stopped         = False
        self.on_event         = self.cfg.get("on_event")

        # ── LTF (low timeframe — örn. 5m) serileri ───────────────
        self.close_series     = {s: deque(maxlen=2048) for s in symbols}
        self.high_series      = {s: deque(maxlen=2048) for s in symbols}
        self.low_series       = {s: deque(maxlen=2048) for s in symbols}
        self.vol_series       = {s: deque(maxlen=2048) for s in symbols}
        self.last_close_time  = {s: 0 for s in symbols}

        # ── HTF (higher timeframe — örn. 1h) serileri ────────────
        self.htf_close_series = {s: deque(maxlen=500) for s in symbols}
        self.htf_high_series  = {s: deque(maxlen=500) for s in symbols}
        self.htf_low_series   = {s: deque(maxlen=500) for s in symbols}
        self.htf_vol_series   = {s: deque(maxlen=500) for s in symbols}
        self.htf_last_time    = {s: 0 for s in symbols}

        self.open_positions    = {}
        self.trade_count_today = 0
        self.pnl_total_usd     = 0.0
        self.daily_pnl_usd     = 0.0
        self._daily_fired      = False
        self.last_reset_day    = time.strftime("%Y-%m-%d")

        # ── Piyasa Rejimi + Sembol Yöneticisi ────────────────────
        self.regime  = MarketRegimeDetector(cfg)
        self.sym_mgr = SymbolManager(cfg, starting_equity=self.equity)

        # ── Kara Liste — dinamik (süreli ban) ────────────────────
        self.blacklist: dict[str, float] = {}

        # ── Kara Liste — config'den kalıcı ban (ÖNCELİK 3) ──────
        bl_cfg = self.cfg.get("symbol_blacklist", {})
        self.blacklist_enabled  = bool(bl_cfg.get("enabled", False))
        _bl_syms                = bl_cfg.get("symbols", []) or []
        self.blacklist_symbols  = set(s.upper() for s in _bl_syms)

        # ── Volatilite Bazlı Pozisyon Filtresi (ÖNCELİK 5) ──────
        vpf = self.cfg.get("vol_position_filter", {})
        self.vpf_enabled       = bool( vpf.get("enabled",           False))
        self.vpf_atr_threshold = float(vpf.get("high_atr_threshold", 2.5))
        self.vpf_size_mult     = float(vpf.get("high_atr_size_mult", 0.5))

        # ── Market Regime / KONSOL Breakout Filtresi ──────────
        mr_cfg = self.cfg.get("market_regime", {})
        self.konsol_breakout_only      = bool( mr_cfg.get("konsol_breakout_only", False))
        self.konsol_min_score          = float(mr_cfg.get("konsol_min_score", 98.0))
        self.konsol_min_adx            = float(mr_cfg.get("konsol_min_adx", 30.0))
        self.konsol_min_atr_pct        = float(mr_cfg.get("konsol_min_atr_pct", 1.2))
        self.konsol_min_vol_ratio      = float(mr_cfg.get("konsol_min_vol_ratio", 1.5))
        self.konsol_min_htf            = float(mr_cfg.get("konsol_min_htf", 70.0))
        self.konsol_rsi_max_long       = float(mr_cfg.get("konsol_rsi_max_long", 68.0))
        self.konsol_size_mult          = float(mr_cfg.get("konsol_size_mult", 0.50))

        # ── Adaptif Sembol Kalite Filtresi ─────────────────────
        sqf = self.cfg.get("symbol_quality_filter", {})
        self.sqf_enabled          = bool( sqf.get("enabled",                True))
        self.sqf_weak_mult        = float(sqf.get("weak_symbol_multiplier", 0.50))
        self.sqf_min_qs           = float(sqf.get("min_qs",                  8.0))
        self.sqf_min_atr_pct      = float(sqf.get("min_atr_pct",             1.2))
        self.sqf_min_adx          = float(sqf.get("min_adx",                20.0))
        self.sqf_score_bonus      = float(sqf.get("score_bonus",             5.0))

        # ── SL Re-entry / Fake Stop Koruması ───────────────────
        re_cfg = self.cfg.get("reentry", {})
        self.reentry_enabled        = bool(re_cfg.get("enabled", False))
        self.reentry_max_per_symbol = int(  re_cfg.get("max_reentries", 1))
        self.reentry_window_bars    = int(  re_cfg.get("window_bars", 4))
        self.reentry_cooldown_bars  = int(  re_cfg.get("cooldown_bars", 1))
        self.reentry_min_score      = float(re_cfg.get("min_score", self.score_long_open))
        self.reentry_min_qs         = float(re_cfg.get("min_qs", 8))
        self.reentry_min_htf        = float(re_cfg.get("min_htf", 75))
        self.reentry_min_adx        = float(re_cfg.get("min_adx", 35))
        self.reentry_size_mult      = float(re_cfg.get("size_mult", 0.75))
        self.reentry_bar_seconds    = int(  re_cfg.get("bar_seconds", 3600))
        self.reentry_candidates     = {}

        self.csv_path          = self.data_dir / "trade_logs.csv"
        self.events_path       = self.data_dir / "engine_events.log"
        self.allowed_symbol    = None

    # ──────────────────────────────────────────────────────────────
    # Durdurma
    # ──────────────────────────────────────────────────────────────
    def stop(self):
        with self.lock:
            self._stopped = True
        self._fire("ENGINE_STOP")
        log_info("Engine durduruldu")

    # ──────────────────────────────────────────────────────────────
    # Event Yayıncısı
    # ──────────────────────────────────────────────────────────────
    def _fire(self, etype: str, **kw):
        ts   = int(time.time())
        line = f"{ts}\t{etype}\t" + " ".join(f"{k}={v}" for k, v in kw.items())
        try:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
        log_event(etype, **kw)
        if callable(self.on_event):
            try:
                self.on_event(etype, kw)
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────
    # Kara Liste
    # ──────────────────────────────────────────────────────────────
    def add_to_blacklist(self, symbol: str, hours: float = 24.0):
        with self.lock:
            self.blacklist[symbol] = time.time() + hours * 3600
        self._fire("BLACKLIST_ADD", symbol=symbol, hours=hours)
        log_info(f"Kara listeye eklendi: {symbol} ({hours}sa)")

    def remove_from_blacklist(self, symbol: str):
        with self.lock:
            self.blacklist.pop(symbol, None)
        log_info(f"Kara listeden çıkarıldı: {symbol}")

    def get_blacklist(self) -> list:
        with self.lock:
            now     = time.time()
            expired = [s for s, exp in self.blacklist.items() if now >= exp]
            for s in expired:
                del self.blacklist[s]
            return [(s, round((exp - now) / 3600, 1))
                    for s, exp in self.blacklist.items()]

    # ──────────────────────────────────────────────────────────────
    # Veri Besleme — LTF
    # ──────────────────────────────────────────────────────────────
    def seed_from_candles(self, symbol: str, candles: list):
        with self.lock:
            d = self.close_series.setdefault(symbol, deque(maxlen=2048))
            h = self.high_series.setdefault( symbol, deque(maxlen=2048))
            l = self.low_series.setdefault(  symbol, deque(maxlen=2048))
            v = self.vol_series.setdefault(  symbol, deque(maxlen=2048))
            for c in candles:
                d.append(float(c.get("close",  0)))
                h.append(float(c.get("high",   0)))
                l.append(float(c.get("low",    0)))
                v.append(float(c.get("volume", 0)))
            if candles:
                self.last_close_time[symbol] = int(candles[-1].get("close_time", 0))

    def on_candle(self, symbol: str, candle: dict):
        with self.lock:
            if self._stopped:
                return
            self._reset_daily_if_needed()
            price  = float(candle.get("close",  0))
            high   = float(candle.get("high",   0))
            low    = float(candle.get("low",    0))
            volume = float(candle.get("volume", 0))
            self.close_series[symbol].append(price)
            self.high_series[symbol].append(high)
            self.low_series[symbol].append(low)
            self.vol_series[symbol].append(volume)
            self.last_close_time[symbol] = int(candle.get("close_time", 0))
            prices  = list(self.close_series[symbol])
            highs   = list(self.high_series[symbol])
            lows    = list(self.low_series[symbol])
            volumes = list(self.vol_series[symbol])
            in_pos  = symbol in self.open_positions

        # ── Lock DIŞINDA CPU-yoğun hesaplama ─────────────────
        if len(prices) < 50:
            return

        news_score = get_sentiment_score()
        result     = score_symbol(prices, highs, lows, volumes, news_score)
        score      = result["final_score"]

        # ── Sonuçla birlikte tekrar lock al ──────────────────
        with self.lock:
            if self._stopped:
                return
            if in_pos:
                self._manage(symbol, price, score)
            else:
                self._try_open(symbol, price, score, prices, volumes, result)

    # ──────────────────────────────────────────────────────────────
    # Veri Besleme — HTF
    # ──────────────────────────────────────────────────────────────
    def seed_from_candles_htf(self, symbol: str, candles: list):
        """1h (veya başka HTF) geçmiş mumlarını yükler."""
        with self.lock:
            pd = self.htf_close_series.setdefault(symbol, deque(maxlen=500))
            hd = self.htf_high_series.setdefault( symbol, deque(maxlen=500))
            ld = self.htf_low_series.setdefault(  symbol, deque(maxlen=500))
            vd = self.htf_vol_series.setdefault(  symbol, deque(maxlen=500))
            for c in candles:
                pd.append(float(c.get("close",  0)))
                hd.append(float(c.get("high",   0)))
                ld.append(float(c.get("low",    0)))
                vd.append(float(c.get("volume", 0)))
            if candles:
                self.htf_last_time[symbol] = int(candles[-1].get("close_time", 0))

    def on_candle_htf(self, symbol: str, candle: dict):
        """1h mumunu HTF serilerine ekler (lock dışından çağrılır)."""
        with self.lock:
            if self._stopped:
                return
            self.htf_close_series[symbol].append(float(candle.get("close",  0)))
            self.htf_high_series[symbol].append( float(candle.get("high",   0)))
            self.htf_low_series[symbol].append(  float(candle.get("low",    0)))
            self.htf_vol_series[symbol].append(  float(candle.get("volume", 0)))

    # ──────────────────────────────────────────────────────────────
    # Günlük Sıfırlama
    # ──────────────────────────────────────────────────────────────
    def _reset_daily_if_needed(self):
        today = time.strftime("%Y-%m-%d")
        if today != self.last_reset_day:
            self.trade_count_today = 0
            self.daily_pnl_usd     = 0.0
            self._daily_fired      = False
            self.last_reset_day    = today
            log_info("Günlük sayaçlar sıfırlandı")

    # ──────────────────────────────────────────────────────────────
    # Pozisyon Büyüklüğü
    # ──────────────────────────────────────────────────────────────
    def _lot(self, price: float, dynamic_sl_pct: float = None) -> float:
        if dynamic_sl_pct is None:
            dynamic_sl_pct = self.sl_pct
        current_equity = self.equity + self.pnl_total_usd
        risk_usdt      = current_equity * self.risk_per_trade
        denom          = price * max(dynamic_sl_pct, 0.001)
        return max(0.0001, risk_usdt / denom)

    # ──────────────────────────────────────────────────────────────
    # Günlük Limit Kontrolleri
    # ──────────────────────────────────────────────────────────────
    def _daily_target_hit(self) -> bool:
        if self._daily_fired:
            return True
        target = (self.equity + self.pnl_total_usd) * self.daily_target_pct
        if self.daily_pnl_usd >= target:
            self._daily_fired = True
            self._fire("DAILY_TARGET_HIT",
                       daily_pnl=round(self.daily_pnl_usd, 2),
                       target=round(target, 2))
            log_info(f"Günlük hedef ulaşıldı: ${self.daily_pnl_usd:.2f}")
            return True
        return False

    def _daily_loss_hit(self) -> bool:
        limit = (self.equity + self.pnl_total_usd) * self.daily_loss_limit
        if self.daily_pnl_usd <= -limit:
            self._fire("DAILY_LOSS_LIMIT",
                       daily_pnl=round(self.daily_pnl_usd, 2),
                       limit=round(-limit, 2))
            log_info(f"Günlük zarar limitine ulaşıldı: ${self.daily_pnl_usd:.2f}")
            return True
        return False

    # ──────────────────────────────────────────────────────────────
    # HTF Skor Hesapla
    # ──────────────────────────────────────────────────────────────
    def _htf_score(self, symbol: str) -> float:
        """
        1h serisinden skor hesaplar.
        Yeterli veri yoksa 50.0 (nötr) döner — bu durumda
        MTF filtresi engel olmaz (fail-open davranışı).
        """
        prices  = list(self.htf_close_series.get(symbol, []))
        highs   = list(self.htf_high_series.get( symbol, []))
        lows    = list(self.htf_low_series.get(  symbol, []))
        volumes = list(self.htf_vol_series.get(  symbol, []))
        if len(prices) < 50:
            return 50.0
        try:
            result = score_symbol(prices, highs, lows, volumes)
            return result["final_score"]
        except Exception:
            return 50.0

    # ──────────────────────────────────────────────────────────────
    # Ana İşlem Akışı
    # ──────────────────────────────────────────────────────────────


    # ── Pozisyon Yönetimi ─────────────────────────────────────────
    def _manage(self, symbol: str, price: float, score: float):
        pos    = self.open_positions[symbol]
        age    = time.time() - pos["ts_open"]
        mult   = 1 if pos["side"] == "LONG" else -1
        change = (price - pos["entry"]) / pos["entry"] * mult

        pos_sl_pct = pos.get("sl_pct", self.sl_pct)
        if change <= -pos_sl_pct:
            self._register_reentry_candidate(symbol, pos, price, reason="SL")
            self._close(symbol, price, change, "SL")
            return

        if self.trail and change > 0:
            # Pozisyona özgü trail_step kullan (açılışta rejime göre belirlendi)
            pos_trail    = pos.get("trail_step", self.trail_step)
            trail_locked = pos.get("trail_locked", None)
            if trail_locked is None or change > trail_locked + pos_trail:
                pos["trail_locked"] = change
                self._fire("TRAIL_LOCK", symbol=symbol,
                           locked=f"{change*100:.2f}%")

        if age < self.min_hold:
            return

        if age >= self.max_hold_sec:
            self._close(symbol, price, change, "MaxHold")
            return

        reason = self._exit_reason(pos, price, change, score)
        if reason:
            self._close(symbol, price, change, reason)

    def _exit_reason(self, pos: dict, price: float, change: float, score: float):
        pos_sl_pct = pos.get("sl_pct", self.sl_pct)
        if change <= -pos_sl_pct:
            return "SL"
        if change >= self.tp_pct and change >= self.min_profit_close:
            return "TP"
        if change >= self.min_profit_close:
            if pos["side"] == "LONG"  and score < self.score_close:
                return "ScoreClose"
            if pos["side"] == "SHORT" and score > self.score_close:
                return "ScoreClose"
            locked = pos.get("trail_locked", change)
            # Pozisyona özgü trail_step (açılışta rejime göre belirlendi)
            pos_trail = pos.get("trail_step", self.trail_step)
            if self.trail and change < locked - pos_trail:
                return "Trail"
        return None

    # ── Pozisyon Açma (v6: HTF konfirmasyon eklendi) ─────────────
    def _trade_quality_score(self, symbol: str, result: dict, side: str) -> int:
        """Canlı engine için backtestteki QS mantığıyla uyumlu hafif setup kalite puanı."""
        comp = result.get("components", {}) or {}
        score = 0
        htf = self._htf_score(symbol)
        if side == "LONG" and htf >= self.mtf_long_min:
            score += 2
        if side == "SHORT" and htf <= self.mtf_short_max:
            score += 2
        atr_pct = float(comp.get("atr_pct", 0.0) or 0.0)
        if 0.3 <= atr_pct <= 3.0:
            score += 2
        regime = self.regime._last_regime
        if regime == "TREND":
            score += 2
        elif regime == "KONSOL":
            score += 1
        vol_ratio = float(comp.get("volume_ratio", comp.get("vol_ratio", 0.0)) or 0.0)
        if vol_ratio >= 1.5:
            score += 2
        elif vol_ratio >= 1.1:
            score += 1
        try:
            sentiment = get_market_sentiment()
            if side == "LONG" and sentiment == "BULLISH":
                score += 2
            if side == "SHORT" and sentiment == "BEARISH":
                score += 2
        except Exception:
            pass
        return max(0, min(10, int(score)))

    def _symbol_quality_filter(self, symbol: str, result: dict, side: str, qs_pts: float, score: float):
        """Kötü performanslı sembolde yalnızca zayıf setup'ı engeller; coin evrenden atılmaz."""
        if not self.sqf_enabled:
            return True, "DISABLED", 1.0, 0.0
        if symbol == "BTCUSDT":
            return True, "BTC_REFERENCE", 1.0, 0.0
        comp = result.get("components", {}) or {}
        atr_pct = float(comp.get("atr_pct", 0.0) or 0.0)
        adx_val = float(comp.get("adx", 0.0) or 0.0)
        sym_mult = float(self.sym_mgr.size_multiplier(symbol))
        rolling_pnl = float(self.sym_mgr.get_rolling_pnl(symbol))
        if sym_mult > self.sqf_weak_mult:
            return True, "SYMBOL_OK", sym_mult, rolling_pnl
        reasons = []
        if qs_pts < self.sqf_min_qs:
            reasons.append(f"qs={qs_pts}<min_qs={self.sqf_min_qs}")
        if atr_pct < self.sqf_min_atr_pct:
            reasons.append(f"atr={atr_pct:.3f}<min_atr={self.sqf_min_atr_pct}")
        if adx_val > 0 and adx_val < self.sqf_min_adx:
            reasons.append(f"adx={adx_val:.1f}<min_adx={self.sqf_min_adx}")
        if side == "LONG" and score < (self.score_long_open + self.sqf_score_bonus):
            reasons.append(f"score={score:.1f}<long+bonus={self.score_long_open + self.sqf_score_bonus:.1f}")
        if side == "SHORT" and self.score_short_open < 100 and score > (self.score_short_open - self.sqf_score_bonus):
            reasons.append(f"score={score:.1f}>short-bonus={self.score_short_open - self.sqf_score_bonus:.1f}")
        if reasons:
            return False, " | ".join(reasons), sym_mult, rolling_pnl
        return True, "WEAK_SYMBOL_BUT_SETUP_OK", sym_mult, rolling_pnl

    def _register_reentry_candidate(self, symbol: str, pos: dict, price: float, reason: str = "SL"):
        """SL sonrası kısa süreli re-entry adayı kaydeder."""
        if not self.reentry_enabled or reason != "SL" or symbol == "BTCUSDT":
            return
        used = int(pos.get("reentry_count", 0))
        if used >= self.reentry_max_per_symbol:
            return
        now = time.time()
        cand = {
            "side": pos.get("side", "LONG"),
            "created_ts": now,
            "earliest_ts": now + self.reentry_cooldown_bars * self.reentry_bar_seconds,
            "expires_ts": now + self.reentry_window_bars * self.reentry_bar_seconds,
            "used": used,
            "entry": pos.get("entry"),
            "sl_price": price,
            "prev_score": pos.get("score"),
            "prev_qs": pos.get("qs_score"),
            "prev_htf": pos.get("htf_score"),
            "prev_adx": pos.get("adx"),
        }
        self.reentry_candidates[symbol] = cand
        self._fire("REENTRY_CANDIDATE", symbol=symbol, side=cand["side"],
                   used=used, cooldown_bars=self.reentry_cooldown_bars,
                   window_bars=self.reentry_window_bars, sl_price=price)

    def _get_reentry_candidate(self, symbol: str, side: str):
        cand = self.reentry_candidates.get(symbol)
        if not self.reentry_enabled or not cand:
            return None
        now = time.time()
        if now > cand.get("expires_ts", 0):
            self.reentry_candidates.pop(symbol, None)
            self._fire("REENTRY_EXPIRED", symbol=symbol, side=side)
            return None
        if side != cand.get("side"):
            return None
        if now < cand.get("earliest_ts", 0):
            return None
        return cand

    def _reentry_ok(self, symbol: str, side: str, score: float, qs_pts: float, htf_score: float, adx_val: float):
        cand = self._get_reentry_candidate(symbol, side)
        if not cand:
            return False, "no_candidate"
        reasons = []
        if score < self.reentry_min_score:
            reasons.append(f"score={score:.1f}<min={self.reentry_min_score:.1f}")
        if qs_pts < self.reentry_min_qs:
            reasons.append(f"qs={qs_pts}<min={self.reentry_min_qs}")
        if htf_score < self.reentry_min_htf:
            reasons.append(f"htf={htf_score:.1f}<min={self.reentry_min_htf:.1f}")
        if adx_val > 0 and adx_val < self.reentry_min_adx:
            reasons.append(f"adx={adx_val:.1f}<min={self.reentry_min_adx:.1f}")
        if reasons:
            self._fire("REENTRY_BLOCKED", symbol=symbol, side=side, detail=" | ".join(reasons),
                       score=round(score, 1), qs_score=qs_pts, htf_score=round(htf_score, 1), adx=round(adx_val, 1))
            return False, " | ".join(reasons)
        return True, "REENTRY_OK"

    def _effective_long_threshold(self, symbol: str, result: dict, base_adsl_threshold: float) -> float:
        if not self.dynamic_threshold_enabled:
            return float(base_adsl_threshold)
        regime = self.regime._last_regime
        if regime == "TREND":
            thr = self.dt_trend_score
        elif regime == "KONSOL":
            thr = self.dt_konsol_score
        elif regime == "BEARISH":
            thr = self.dt_bearish_score
        else:
            thr = self.dt_neutral_score
        comp = result.get("components", {}) or {}
        adx_val = float(comp.get("adx", 0.0) or 0.0)
        rsi_val = float(comp.get("rsi", 50.0) or 50.0)
        htf_val = float(self._htf_score(symbol)) if self.mtf_enabled else 100.0
        if htf_val >= self.dt_strong_htf and adx_val >= self.dt_strong_adx and rsi_val <= self.dt_strong_rsi_max:
            thr -= self.dt_strong_discount
        return max(self.dt_min_score, min(self.dt_max_score, float(thr)))

    def _konsol_breakout_ok(self, symbol: str, result: dict, score: float, htf_score: float, volumes: list, side: str) -> bool:
        if not self.konsol_breakout_only or self.regime._last_regime != "KONSOL":
            return True
        comp = result.get("components", {}) or {}
        adx_val = float(comp.get("adx", 0.0) or 0.0)
        atr_pct = float(comp.get("atr_pct", 0.0) or 0.0)
        rsi_val = float(comp.get("rsi", 50.0) or 50.0)
        vol_ratio = 0.0
        if len(volumes) >= 20:
            base = sum(volumes[-20:-1]) / 19
            vol_ratio = (volumes[-1] / base) if base > 0 else 0.0
        checks = [
            (score >= self.konsol_min_score, "KONSOL_LOW_SCORE", f"score={score:.1f}<min={self.konsol_min_score:.1f}"),
            (adx_val >= self.konsol_min_adx, "KONSOL_WEAK_ADX", f"adx={adx_val:.1f}<min={self.konsol_min_adx:.1f}"),
            (atr_pct >= self.konsol_min_atr_pct, "KONSOL_LOW_ATR", f"atr={atr_pct:.3f}<min={self.konsol_min_atr_pct:.3f}"),
            (vol_ratio >= self.konsol_min_vol_ratio, "KONSOL_NO_VOLUME_BREAKOUT", f"vol_ratio={vol_ratio:.2f}<min={self.konsol_min_vol_ratio:.2f}"),
            (htf_score >= self.konsol_min_htf, "KONSOL_HTF_WEAK", f"htf={htf_score:.1f}<min={self.konsol_min_htf:.1f}"),
        ]
        if side == "LONG":
            checks.append((rsi_val <= self.konsol_rsi_max_long, "KONSOL_RSI_OVERHEATED", f"rsi={rsi_val:.1f}>max={self.konsol_rsi_max_long:.1f}"))
        for ok, cause, detail in checks:
            if not ok:
                self._fire("OPEN_BLOCK", cause=cause, symbol=symbol, score=round(score, 1), detail=detail,
                           adx=round(adx_val, 2), atr=round(atr_pct, 3), rsi=round(rsi_val, 1),
                           htf_score=round(htf_score, 1), vol_ratio=round(vol_ratio, 2), side=side)
                return False
        return True

    def _try_open(self, symbol: str, price: float, score: float,
                  prices: list, volumes: list, result: dict):
        # Temel kontrol kapıları
        if self._stopped:                                         return
        if len(self.open_positions) >= self.max_open_pos:         return
        if self.trade_count_today  >= self.max_trades_day:        return
        if self.allowed_symbol and symbol != self.allowed_symbol: return
        if self._daily_target_hit():                              return
        if self._daily_loss_hit():                                return

        # Kara liste kontrolü — dinamik (süreli ban)
        bl_exp = self.blacklist.get(symbol)
        if bl_exp is not None:
            if time.time() < bl_exp:
                return
            else:
                del self.blacklist[symbol]

        # Kara liste kontrolü — config'den kalıcı ban (ÖNCELİK 3)
        # BTCUSDT referans filtreleri beslediği için kalıcı blacklist ona uygulanmaz.
        if symbol != "BTCUSDT" and self.blacklist_enabled and symbol in self.blacklist_symbols:
            self._fire("OPEN_BLOCK", cause="SYMBOL_BLACKLIST", symbol=symbol)
            return

        # Minimum işlem hacmi filtresi
        if len(prices) >= 20:
            avg_notional = (sum(prices[-20:]) / 20) * (sum(volumes[-20:]) / 20)
            if avg_notional < self.min_notional:
                return

        # Hacim patlaması filtresi
        if len(volumes) >= 20:
            rv = sum(volumes[-3:]) / 3
            hv = sum(volumes[-20:-3]) / 17
            if hv > 0 and rv < hv * self.vol_mult:
                self._fire("OPEN_BLOCK", cause="LOW_VOLUME", symbol=symbol)
                return

        # Portföy yönü filtresi
        btc = self.open_positions.get("BTCUSDT")
        if btc:
            if btc["side"] == "SHORT" and score >= self.score_long_open:
                self._fire("OPEN_BLOCK", cause="BTC_SHORT_BLOCKS_LONG", symbol=symbol)
                return
            if btc["side"] == "LONG" and score <= self.score_short_open:
                if score > self.score_short_open / 2:
                    self._fire("OPEN_BLOCK", cause="BTC_LONG_WEAK_SHORT", symbol=symbol)
                    return

        # Makro sentiment filtresi
        sentiment = get_market_sentiment()
        # Adaptif giriş eşiği: KONSOL'de +7, BEARISH'de +99 (pratikte giriş yok)
        _adsl_thr = adaptive_sl.compute(
            regime               = self.regime._last_regime,
            atr_pct              = result.get("components", {}).get("atr_pct", 0.0),
            base_score_threshold = self.score_long_open,
            base_atr_multiplier  = self.atr_multiplier,
            base_trail_step      = self.trail_step,
            cfg                  = self.cfg,
        )
        effective_long_thr = self._effective_long_threshold(symbol, result, _adsl_thr["score_threshold"])
        side      = None
        if score >= effective_long_thr and sentiment != "BEARISH":
            side = "LONG"
        elif score <= self.score_short_open and sentiment != "BULLISH":
            side = "SHORT"

        if not side:
            return
        adx_val = result.get("components", {}).get("adx", 0.0)
        if self.adx_filter_enabled and adx_val > 0 and adx_val < self.adx_filter_threshold:
            return

        # ── ATR Minimum Filtresi ───────────────────────────────
        if self.atr_filter_enabled:
            atr_val = result.get("components", {}).get("atr_pct", 0.0)
            if atr_val < self.atr_filter_min:
                self._fire("OPEN_BLOCK", cause="ATR_TOO_LOW",
                           symbol=symbol, atr=round(atr_val, 3))
                return

        # ── RSI Filtresi ───────────────────────────────────────
        if self.rsi_filter_enabled:
            rsi_val = result.get("components", {}).get("rsi", 50.0)
            if side == "LONG"  and rsi_val > self.rsi_max_long:
                self._fire("OPEN_BLOCK", cause="RSI_TOO_HIGH",
                           symbol=symbol, rsi=round(rsi_val, 1))
                return
            if side == "SHORT" and rsi_val < self.rsi_min_short:
                self._fire("OPEN_BLOCK", cause="RSI_TOO_LOW",
                           symbol=symbol, rsi=round(rsi_val, 1))
                return

        htf_sc = self._htf_score(symbol) if self.mtf_enabled else 100.0
        if not self._konsol_breakout_ok(symbol, result, score, htf_sc, volumes, side):
            return

        # ── Çoklu Timeframe Konfirmasyon ─────────────────────────
        if self.mtf_enabled:
            if side == "LONG"  and htf_sc < self.mtf_long_min:
                self._fire("OPEN_BLOCK", cause="MTF_NO_CONFIRM_LONG",
                           symbol=symbol, htf_score=round(htf_sc, 1))
                return
            if side == "SHORT" and htf_sc > self.mtf_short_max:
                self._fire("OPEN_BLOCK", cause="MTF_NO_CONFIRM_SHORT",
                           symbol=symbol, htf_score=round(htf_sc, 1))
                return

        # Adaptif sembol kalite filtresi: sembolü listeden atmaz,
        # son dönemde kötü çalışan sembolde zayıf setup'ı engeller.
        qs_pts = self._trade_quality_score(symbol, result, side)
        sq_ok, sq_detail, sym_mult, rolling_pnl = self._symbol_quality_filter(symbol, result, side, qs_pts, score)
        if not sq_ok:
            comp = result.get("components", {}) or {}
            self._fire(
                "OPEN_BLOCK",
                cause="WEAK_SYMBOL_LOW_QUALITY",
                symbol=symbol,
                side=side,
                score=round(score, 1),
                qs_score=qs_pts,
                symbol_mult=round(sym_mult, 3),
                rolling_pnl=round(rolling_pnl, 3),
                atr=round(float(comp.get("atr_pct", 0.0) or 0.0), 3),
                adx=round(float(comp.get("adx", 0.0) or 0.0), 2),
                detail=sq_detail,
            )
            return

        htf_for_reentry = self._htf_score(symbol) if self.mtf_enabled else 100.0
        is_reentry, reentry_detail = self._reentry_ok(symbol, side, score, qs_pts, htf_for_reentry, adx_val)
        final_size_mult = sym_mult * (self.reentry_size_mult if is_reentry else 1.0)
        if self.konsol_breakout_only and self.regime._last_regime == "KONSOL":
            final_size_mult *= self.konsol_size_mult
        self._open(symbol, price, side, result, size_mult=final_size_mult, is_reentry=is_reentry, reentry_detail=reentry_detail)

    # ──────────────────────────────────────────────────────────────
    # Pozisyon Aç / Kapat
    # ──────────────────────────────────────────────────────────────
    def _open(self, symbol: str, price: float, side: str, result: dict, size_mult: float = 1.0, is_reentry: bool = False, reentry_detail: str = ""):
        # Adaptif SL: ATR × rejime özgü çarpan
        atr_pct_val = result.get("components", {}).get("atr_pct", 0.0)
        regime      = self.regime._last_regime
        adsl = adaptive_sl.compute(
            regime               = regime,
            atr_pct              = atr_pct_val,
            base_score_threshold = self.score_long_open,
            base_atr_multiplier  = self.atr_multiplier,
            base_trail_step      = self.trail_step,
            cfg                  = self.cfg,
        )
        if self.use_atr_stop and atr_pct_val > 0:
            final_sl_pct = adsl["sl_pct"]
        else:
            final_sl_pct = self.sl_pct
        pos_trail_step = adsl["trail_step"] if self.dynamic_trail_enabled else self.trail_step

        qty = self._lot(price, dynamic_sl_pct=final_sl_pct) * size_mult

        # ── Volatilite Bazlı Pozisyon Küçültme (ÖNCELİK 5) ───
        if self.vpf_enabled and atr_pct_val > self.vpf_atr_threshold:
            qty *= self.vpf_size_mult
            self._fire("VOL_SIZE_REDUCED",
                       symbol=symbol,
                       atr=round(atr_pct_val, 2),
                       mult=self.vpf_size_mult)

        self.open_positions[symbol] = {
            "side":       side,
            "entry":      price,
            "qty":        qty,
            "ts_open":    time.time(),
            "sl_pct":     final_sl_pct,
            "trail_step": pos_trail_step,   # rejime göre belirlendi
            "score":      result.get("final_score", 0.0),
            "qs_score":   self._trade_quality_score(symbol, result, side),
            "htf_score":  self._htf_score(symbol) if self.mtf_enabled else 0.0,
            "adx":        result.get("components", {}).get("adx", 0.0),
            "is_reentry": int(is_reentry),
            "reentry_count": int((self.reentry_candidates.get(symbol) or {}).get("used", 0)) + (1 if is_reentry else 0),
        }
        self.trade_count_today += 1
        self._log_trade(symbol, side, qty, price, "", 0.0, 0.0,
                        f"OPEN sl_pct={final_sl_pct*100:.2f}% regime={regime} score={result['final_score']}")
        if is_reentry:
            self.reentry_candidates.pop(symbol, None)
            self._fire("REENTRY_OPEN", symbol=symbol, side=side, entry=price, detail=reentry_detail,
                       score=result.get("final_score", 0.0))
        self._fire("OPEN", symbol=symbol, side=side,
                   entry=price, score=result["final_score"], regime=regime, is_reentry=int(is_reentry))

    def _close(self, symbol: str, price: float, change_pct: float, reason: str):
        pos = self.open_positions.pop(symbol, None)
        if not pos:
            return
        entry   = pos["entry"]
        qty     = pos["qty"]
        pnl_usd = ((price - entry) if pos["side"] == "LONG"
                   else (entry - price)) * qty

        self.pnl_total_usd += pnl_usd
        self.daily_pnl_usd += pnl_usd
        self.sym_mgr.record_trade(symbol, pnl_usd)
        self.sym_mgr.update_equity(self.equity + self.pnl_total_usd)

        self._log_trade(symbol, pos["side"], qty, entry, price,
                        round(change_pct * 100, 3), round(pnl_usd, 3), reason)
        self._fire("EXIT", symbol=symbol, side=pos["side"], reason=reason,
                   pnl_usd=round(pnl_usd, 2),
                   pnl_pct=f"{change_pct*100:.2f}%",
                   is_reentry=pos.get("is_reentry", 0),
                   reentry_count=pos.get("reentry_count", 0))

    # ──────────────────────────────────────────────────────────────
    # CSV Log
    # ──────────────────────────────────────────────────────────────
    def _log_trade(self, sym, side, qty, entry, exitp,
                   kar_pct, kar_usd, note):
        ts  = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        try:
            with open(self.csv_path, "a", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f, delimiter=";")
                if new:
                    w.writerow(["Tarih","Sembol","Yon","GirisFiyati",
                                "CikisFiyati","KarYuzde","KarUSD","Not"])
                w.writerow([ts, sym, side, entry or "",
                            exitp or "", kar_pct, kar_usd, note])
        except Exception as e:
            log_error(f"CSV yazma hatası: {e}")

    # ──────────────────────────────────────────────────────────────
    # GUI'ye Veri
    # ──────────────────────────────────────────────────────────────
    def get_open_positions(self) -> list:
        with self.lock:
            out = []
            for sym, p in self.open_positions.items():
                age = int(time.time() - p["ts_open"])
                out.append({
                    "symbol":    sym,
                    "side":      p["side"],
                    "entry":     p["entry"],
                    "age_min":   round(age / 60, 1),
                    "trail_pct": round(p.get("trail_locked", 0.0) * 100, 2),
                })
            return out

    def get_pnl(self) -> dict:
        with self.lock:
            base = self.equity
            return {
                "usd":       round(self.pnl_total_usd, 2),
                "pct":       round(self.pnl_total_usd / base * 100, 3) if base else 0.0,
                "daily_usd": round(self.daily_pnl_usd, 2),
                "equity":    round(base + self.pnl_total_usd, 2),
            }

    def set_allowed_symbol(self, sym_or_none):
        with self.lock:
            self.allowed_symbol = sym_or_none