# market_regime.py — BTC Piyasa Rejimi Motoru (Modül 1)
#
# DÜZELTMELER (v1.1):
# BUG-8: Rejim kararı mantık çakışması düzeltildi.
#   Eski: full_trend+atr_chaotic+vol_strong → TREND (kaotik ATR'de risk)
#   Yeni: full_trend+atr_chaotic → KONSOL (kaotik ATR güvenilmez)
#         full_trend+atr_chaotic+vol_strong → KONSOL (hacim olsa da kaotik)
#   Sadece full_trend+NOT atr_chaotic → TREND
#
# BUG-4 (bilgi): EMA10/21/55 + min_candles=70 config değerleri config'den
#   doğru okunuyor. min_candles=70 ile EMA55 hesabı marjinal ama çalışıyor.
#   full_trend koşulunun nadiren sağlanması normal — bu config tercihidir.

import numpy as np
import pandas as pd


class MarketRegimeDetector:
    """
    BTC fiyat serisinden anlık piyasa rejimini tespit eder.
    """

    TREND   = "TREND"
    KONSOL  = "KONSOL"
    BEARISH = "BEARISH"

    SIZE = {TREND: 1.0, KONSOL: 0.5, BEARISH: 0.0}

    def __init__(self, cfg: dict):
        r = cfg.get("market_regime", {})
        self.enabled          = bool( r.get("enabled",         True))
        self.ema_fast         = int(  r.get("ema_fast",          20))
        self.ema_slow         = int(  r.get("ema_slow",          50))
        self.ema_long         = int(  r.get("ema_long",         200))
        self.atr_period       = int(  r.get("atr_period",        14))
        self.atr_mult_thresh  = float(r.get("atr_mult_thresh",  1.8))
        self.vol_window       = int(  r.get("vol_window",        20))
        self.vol_burst_mult   = float(r.get("vol_burst_mult",   1.5))
        self.min_candles      = int(  r.get("min_candles",      210))

        self._last_regime = self.KONSOL
        self._last_detail = {}

    def detect(self, btc_closes: list,
               btc_highs: list = None,
               btc_lows:  list = None,
               btc_vols:  list = None) -> str:

        if not self.enabled:
            return self.TREND

        if len(btc_closes) < self.min_candles:
            return self._last_regime

        arr   = np.array(btc_closes, dtype=float)
        s     = pd.Series(arr)
        price = arr[-1]

        # ── Katman 1: EMA trend ──────────────────────────────
        ema_f = float(s.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1])
        ema_s = float(s.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1])
        ema_l = float(s.ewm(span=self.ema_long, adjust=False).mean().iloc[-1])

        full_trend   = price > ema_f > ema_s > ema_l
        weak_trend   = price > ema_s > ema_l
        bearish_ema  = price < ema_l

        # ── Katman 2: ATR volatilite ─────────────────────────
        atr_chaotic  = False
        atr_percentile = 50.0

        if btc_highs and btc_lows and len(btc_highs) >= self.atr_period + 1:
            h = np.array(btc_highs[-self.atr_period*3:], dtype=float)
            l = np.array(btc_lows[ -self.atr_period*3:], dtype=float)
            c = arr[-self.atr_period*3:]
            trs = []
            for i in range(1, len(c)):
                tr = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
                trs.append(tr)
            if len(trs) >= self.atr_period * 2:
                recent_atr = float(np.mean(trs[-self.atr_period:]))
                base_atr   = float(np.mean(trs[-self.atr_period*2:-self.atr_period]))
                if base_atr > 0:
                    atr_chaotic = recent_atr / base_atr > self.atr_mult_thresh
                if len(trs) >= 20:
                    window = trs[-100:] if len(trs) >= 100 else trs
                    atr_percentile = float(
                        sum(1 for x in window if x <= recent_atr) / len(window) * 100
                    )

        # ── Katman 3: Hacim momentum ─────────────────────────
        vol_strong = False
        if btc_vols and len(btc_vols) >= self.vol_window + 3:
            recent_vol = float(np.mean(btc_vols[-3:]))
            base_vol   = float(np.mean(btc_vols[-self.vol_window:-3]))
            if base_vol > 0:
                vol_strong = recent_vol / base_vol >= self.vol_burst_mult

        # ── Rejim kararı ─────────────────────────────────────
        # BUG-8 DÜZELTMESİ:
        # Eski mantıkta: full_trend + atr_chaotic + vol_strong → TREND
        # Bu kaotik bir piyasada (yüksek ATR volatilite) TREND vermek riskli.
        # Düzeltme: TREND yalnızca full_trend VE NOT atr_chaotic durumunda.
        # Kaotik ATR varsa, hacim ne kadar güçlü olursa olsun, KONSOL.

        if bearish_ema:
            regime = self.BEARISH
        elif full_trend and not atr_chaotic:
            # Normal trend: EMA sırası tam + volatilite normal
            regime = self.TREND
        elif atr_percentile < 10:
            # Gerçek sıkışma: çok düşük ATR
            regime = self.KONSOL
        elif weak_trend:
            # Zayıf trend veya full_trend+atr_chaotic: güvenli taraf KONSOL
            regime = self.KONSOL
        else:
            regime = self.KONSOL

        self._last_regime = regime
        self._last_detail = {
            "regime":         regime,
            "price":          round(price, 2),
            "ema_fast":       round(ema_f, 2),
            "ema_slow":       round(ema_s, 2),
            "ema_long":       round(ema_l, 2),
            "full_trend":     full_trend,
            "weak_trend":     weak_trend,
            "bearish_ema":    bearish_ema,
            "atr_chaotic":    atr_chaotic,
            "vol_strong":     vol_strong,
            "atr_percentile": round(atr_percentile, 1),
        }
        return regime

    def size_multiplier(self) -> float:
        return self.SIZE.get(self._last_regime, 0.5)

    def is_open(self) -> bool:
        return self._last_regime in (self.TREND, self.KONSOL)

    def detail(self) -> dict:
        return self._last_detail.copy()

    def snapshot(self) -> dict:
        return {
            "enabled":         self.enabled,
            "regime":          self._last_regime,
            "size_multiplier": self.size_multiplier(),
        }
