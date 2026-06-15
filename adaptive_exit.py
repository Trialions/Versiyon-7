# adaptive_exit.py — V7 Adaptive Exit Shadow Classifier v2.0 SAFE
# ------------------------------------------------------------
# v2 düzeltmeleri:
# - TREND_RUNNER için sadece skor değil, gate şartları da gerekir.
# - KONSOL/CHOP/RANGE rejimde TREND_RUNNER kesin yasak.
# - btc_trend_ok bilinmiyorsa pozitif bonus verilmez; False ise sert ceza alır.
# - HTF dominantlığı azaltıldı.
# - Symbol behavior düşük örneklemde etkisiz kalır.
# - Bu modül live ve backtest tarafından ortak kullanılmak üzere bağımsızdır.

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


CHOP_REGIMES = {"KONSOL", "CONSOLIDATION", "CHOP", "RANGE", "SIDEWAYS"}
TREND_REGIMES = {"TREND", "BULL", "BULLISH", "MOMENTUM", "UPTREND"}
RISK_REGIMES = {"BEAR", "BEARISH", "HIGH_VOL_RISK", "RISK_OFF"}


@dataclass(frozen=True)
class ExitPolicy:
    name: str
    tp1_close_pct: float
    trail_step_pct: float
    max_hold_hours: Optional[float]
    breakeven_after_r: float
    breakeven_buffer_pct: float
    allow_trend_runner: bool
    size_mult: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveExitDecision:
    enabled: bool
    shadow_mode: bool
    trade_class: str
    continuation_score: float
    confidence: float
    policy: ExitPolicy
    reasons: str
    regime: str
    symbol_behavior_score: float
    symbol_sample_weight: float

    @property
    def policy_name(self) -> str:
        return self.policy.name

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["policy_name"] = self.policy.name
        return d


DEFAULT_ADAPTIVE_EXIT_CFG: Dict[str, Any] = {
    "enabled": True,
    "shadow_mode": True,
    "min_confidence": 55,
    "trend_runner_min_score": 78,
    "scalp_exit_max_score": 43,
    "exhausted_rsi": 72,
    "strict_chop_guard": True,
    "chop_regimes": sorted(CHOP_REGIMES),
    "trend_regimes": sorted(TREND_REGIMES),
    "risk_regimes": sorted(RISK_REGIMES),
    "trend_runner_gates": {
        "min_entry_score": 95,
        "min_htf_score": 70,
        "min_adx": 22,
        "max_rsi": 70,
        "min_vol_ratio": 0.85,
        "require_btc_ok": True,
    },
    "symbol_behavior": {
        "enabled": True,
        "min_closed_trades": 25,
        "full_weight_trades": 50,
        "min_ghost_samples": 100,
        "max_score_impact": 6,
        "neutral_score": 50,
    },
    "policies": {
        "TREND_RUNNER": {
            "tp1_close_pct": 0.08,
            "trail_step_pct": 2.20,
            "max_hold_hours": 72,
            "breakeven_after_r": 1.0,
            "breakeven_buffer_pct": 0.25,
            "allow_trend_runner": True,
            "size_mult": 1.0,
        },
        "NORMAL": {
            "tp1_close_pct": 0.10,
            "trail_step_pct": 1.20,
            "max_hold_hours": 48,
            "breakeven_after_r": 1.0,
            "breakeven_buffer_pct": 0.15,
            "allow_trend_runner": False,
            "size_mult": 1.0,
        },
        "SCALP_EXIT": {
            "tp1_close_pct": 0.45,
            "trail_step_pct": 0.75,
            "max_hold_hours": 18,
            "breakeven_after_r": 0.80,
            "breakeven_buffer_pct": 0.12,
            "allow_trend_runner": False,
            "size_mult": 1.0,
        },
        "CHOP_RISK": {
            "tp1_close_pct": 0.50,
            "trail_step_pct": 0.60,
            "max_hold_hours": 12,
            "breakeven_after_r": 0.70,
            "breakeven_buffer_pct": 0.10,
            "allow_trend_runner": False,
            "size_mult": 0.75,
        },
        "EXHAUSTED": {
            "tp1_close_pct": 0.60,
            "trail_step_pct": 0.60,
            "max_hold_hours": 12,
            "breakeven_after_r": 0.70,
            "breakeven_buffer_pct": 0.10,
            "allow_trend_runner": False,
            "size_mult": 0.75,
        },
        "DISABLED": {
            "tp1_close_pct": 0.10,
            "trail_step_pct": 1.20,
            "max_hold_hours": 48,
            "breakeven_after_r": 1.0,
            "breakeven_buffer_pct": 0.15,
            "allow_trend_runner": False,
            "size_mult": 1.0,
        },
    },
}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def get_adaptive_exit_config(cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    user = (cfg or {}).get("adaptive_exit", {}) or {}
    return _deep_merge(DEFAULT_ADAPTIVE_EXIT_CFG, user)


def _float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _norm_regime(regime: Any) -> str:
    return str(regime or "UNKNOWN").upper()


def _rsi_from_closes(closes: Iterable[float], period: int = 14) -> float:
    vals = [float(x) for x in list(closes or [])]
    if len(vals) <= period:
        return 50.0
    gains, losses = [], []
    for i in range(-period, 0):
        delta = vals[i] - vals[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(abs(min(delta, 0.0)))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr_pct(highs: Iterable[float], lows: Iterable[float], closes: Iterable[float], period: int = 14) -> float:
    h = [float(x) for x in list(highs or [])]
    l = [float(x) for x in list(lows or [])]
    c = [float(x) for x in list(closes or [])]
    n = min(len(h), len(l), len(c))
    if n <= period or not c or c[-1] <= 0:
        return 0.0
    trs = []
    for i in range(n - period, n):
        prev_close = c[i - 1] if i > 0 else c[i]
        trs.append(max(h[i] - l[i], abs(h[i] - prev_close), abs(l[i] - prev_close)))
    return (sum(trs) / len(trs)) / c[-1] * 100.0


def _adx_simple(highs: Iterable[float], lows: Iterable[float], closes: Iterable[float], period: int = 14) -> float:
    h = [float(x) for x in list(highs or [])]
    l = [float(x) for x in list(lows or [])]
    c = [float(x) for x in list(closes or [])]
    n = min(len(h), len(l), len(c))
    if n <= period + 1:
        return 0.0
    trs, plus_dm, minus_dm = [], [], []
    for i in range(n - period, n):
        up_move = h[i] - h[i - 1]
        down_move = l[i - 1] - l[i]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        prev_close = c[i - 1]
        trs.append(max(h[i] - l[i], abs(h[i] - prev_close), abs(l[i] - prev_close)))
    tr_sum = sum(trs)
    if tr_sum <= 0:
        return 0.0
    plus_di = 100.0 * sum(plus_dm) / tr_sum
    minus_di = 100.0 * sum(minus_dm) / tr_sum
    denom = plus_di + minus_di
    if denom <= 0:
        return 0.0
    return 100.0 * abs(plus_di - minus_di) / denom


def _volume_ratio(volumes: Iterable[float], short_n: int = 3, long_n: int = 20) -> float:
    vals = [float(x) for x in list(volumes or [])]
    if len(vals) < long_n:
        return 1.0
    short = sum(vals[-short_n:]) / max(short_n, 1)
    long = sum(vals[-long_n:-short_n]) / max(long_n - short_n, 1)
    return short / long if long > 0 else 1.0


def _symbol_score(symbol_stats: Optional[Mapping[str, Any]], cfg: Mapping[str, Any]) -> Tuple[float, float, str]:
    sb_cfg = cfg.get("symbol_behavior", {}) or {}
    if not sb_cfg.get("enabled", True):
        return 50.0, 0.0, "symbol_behavior_disabled"
    stats = dict(symbol_stats or {})
    closed = int(_float(stats.get("closed_trades", stats.get("trades", 0)), 0))
    ghosts = int(_float(stats.get("ghost_samples", 0), 0))
    min_closed = int(sb_cfg.get("min_closed_trades", 25))
    full = max(int(sb_cfg.get("full_weight_trades", 50)), min_closed)
    min_ghost = int(sb_cfg.get("min_ghost_samples", 100))
    if closed < min_closed and ghosts < min_ghost:
        return float(sb_cfg.get("neutral_score", 50)), 0.0, f"low_sample closed={closed} ghost={ghosts}"

    winrate = _float(stats.get("winrate", stats.get("win_rate", 0.5)), 0.5)
    tp_cont = _float(stats.get("tp_continuation_rate", 0.5), 0.5)
    sl_recovery = _float(stats.get("sl_recovery_rate", 0.5), 0.5)
    ghost_edge = _float(stats.get("ghost_tp_sl_edge", 0.0), 0.0)
    raw = 50.0
    raw += (winrate - 0.50) * 30.0
    raw += (tp_cont - 0.50) * 22.0
    raw += (sl_recovery - 0.50) * 8.0
    raw += ghost_edge * 8.0
    raw = max(0.0, min(100.0, raw))
    weight = min(1.0, max(closed / float(full), ghosts / float(max(min_ghost, 1))))
    return raw, weight, f"closed={closed} ghost={ghosts} weight={weight:.2f}"


def _policy_from_cfg(name: str, cfg: Mapping[str, Any]) -> ExitPolicy:
    policies = cfg.get("policies", {}) or {}
    p = dict(policies.get(name, policies.get("NORMAL", {})))
    return ExitPolicy(
        name=name,
        tp1_close_pct=_float(p.get("tp1_close_pct"), 0.10),
        trail_step_pct=_float(p.get("trail_step_pct"), 1.20),
        max_hold_hours=None if p.get("max_hold_hours") is None else _float(p.get("max_hold_hours"), 48.0),
        breakeven_after_r=_float(p.get("breakeven_after_r"), 1.0),
        breakeven_buffer_pct=_float(p.get("breakeven_buffer_pct"), 0.15),
        allow_trend_runner=bool(p.get("allow_trend_runner", False)),
        size_mult=_float(p.get("size_mult"), 1.0),
    )


def classify_trade(
    *,
    symbol: str,
    side: str = "LONG",
    score: float = 0.0,
    htf_score: float = 50.0,
    regime: str = "UNKNOWN",
    components: Optional[Mapping[str, Any]] = None,
    cfg: Optional[Mapping[str, Any]] = None,
    prices: Optional[Iterable[float]] = None,
    highs: Optional[Iterable[float]] = None,
    lows: Optional[Iterable[float]] = None,
    volumes: Optional[Iterable[float]] = None,
    current_r: float = 0.0,
    btc_trend_ok: Optional[bool] = None,
    symbol_stats: Optional[Mapping[str, Any]] = None,
) -> AdaptiveExitDecision:
    ae_cfg = get_adaptive_exit_config(cfg or {})
    enabled = bool(ae_cfg.get("enabled", True))
    shadow = bool(ae_cfg.get("shadow_mode", True))
    regime_u = _norm_regime(regime)
    comp = dict(components or {})

    if not enabled:
        return AdaptiveExitDecision(False, shadow, "DISABLED", 50.0, 0.0, _policy_from_cfg("DISABLED", ae_cfg), "adaptive_exit_disabled", regime_u, 50.0, 0.0)

    closes = list(prices or [])
    hi = list(highs or [])
    lo = list(lows or [])
    vol = list(volumes or [])

    rsi = _float(comp.get("rsi"), _rsi_from_closes(closes))
    adx = _float(comp.get("adx"), _adx_simple(hi, lo, closes))
    atrp = _float(comp.get("atr_pct"), _atr_pct(hi, lo, closes))
    vol_ratio = _float(comp.get("volume_ratio", comp.get("vol_ratio")), _volume_ratio(vol))
    score = _float(score, _float(comp.get("score"), 0.0))
    htf_score = _float(htf_score, 50.0)
    current_r = _float(current_r, 0.0)

    reasons = []
    cont = 45.0

    # Entry score katkısı: yüksek eşikli V7 için dominant değil, sadece destek.
    if score >= 98:
        cont += 8; reasons.append("score>=98 +8")
    elif score >= 95:
        cont += 5; reasons.append("score>=95 +5")
    elif score >= 90:
        cont += 2; reasons.append("score>=90 +2")
    else:
        cont -= 8; reasons.append("score<90 -8")

    # HTF katkısı düşürüldü; tek başına TREND_RUNNER yapamaz.
    if htf_score >= 85:
        cont += 10; reasons.append("htf>=85 +10")
    elif htf_score >= 70:
        cont += 7; reasons.append("htf>=70 +7")
    elif htf_score >= 55:
        cont += 2; reasons.append("htf>=55 +2")
    else:
        cont -= 12; reasons.append("htf<55 -12")

    if adx >= 35:
        cont += 12; reasons.append("adx>=35 +12")
    elif adx >= 28:
        cont += 8; reasons.append("adx>=28 +8")
    elif adx >= 22:
        cont += 4; reasons.append("adx>=22 +4")
    elif adx > 0:
        cont -= 8; reasons.append("adx_weak -8")

    if 50 <= rsi <= 66:
        cont += 9; reasons.append("rsi_healthy +9")
    elif 66 < rsi <= 70:
        cont += 2; reasons.append("rsi_warm +2")
    elif rsi > 72:
        cont -= 22; reasons.append("rsi_exhausted -22")
    elif rsi < 45:
        cont -= 8; reasons.append("rsi_low -8")

    if vol_ratio >= 1.6:
        cont += 8; reasons.append("vol>=1.6 +8")
    elif vol_ratio >= 1.15:
        cont += 4; reasons.append("vol>=1.15 +4")
    elif vol_ratio < 0.85:
        cont -= 6; reasons.append("vol_dry -6")

    if 1.0 <= atrp <= 3.5:
        cont += 5; reasons.append("atr_ok +5")
    elif atrp > 5.0:
        cont -= 10; reasons.append("atr_too_high -10")
    elif 0 < atrp < 0.7:
        cont -= 8; reasons.append("atr_low -8")

    if btc_trend_ok is True:
        cont += 5; reasons.append("btc_ok +5")
    elif btc_trend_ok is False:
        cont -= 18; reasons.append("btc_bad -18")
    else:
        reasons.append("btc_unknown +0")

    chop_regimes = {str(x).upper() for x in ae_cfg.get("chop_regimes", CHOP_REGIMES)}
    trend_regimes = {str(x).upper() for x in ae_cfg.get("trend_regimes", TREND_REGIMES)}
    risk_regimes = {str(x).upper() for x in ae_cfg.get("risk_regimes", RISK_REGIMES)}
    is_chop = regime_u in chop_regimes
    if regime_u in trend_regimes:
        cont += 8; reasons.append("regime_trend +8")
    elif is_chop:
        cont -= 30; reasons.append("regime_chop_guard -30")
    elif regime_u in risk_regimes:
        cont -= 18; reasons.append("regime_risk -18")

    if current_r >= 2.0:
        cont += 4; reasons.append("current_r>=2 +4")
    elif current_r >= 1.0:
        cont += 2; reasons.append("current_r>=1 +2")

    sym_score, sym_weight, sym_reason = _symbol_score(symbol_stats, ae_cfg)
    max_impact = _float((ae_cfg.get("symbol_behavior", {}) or {}).get("max_score_impact"), 6.0)
    sym_impact = ((sym_score - 50.0) / 50.0) * max_impact * sym_weight
    if abs(sym_impact) > 0.01:
        cont += sym_impact
        reasons.append(f"symbol_behavior {sym_impact:+.1f}")
    else:
        reasons.append(sym_reason)

    cont = max(0.0, min(100.0, cont))

    gates = ae_cfg.get("trend_runner_gates", {}) or {}
    trend_gate_ok = True
    gate_failures = []
    if is_chop:
        trend_gate_ok = False; gate_failures.append("chop")
    if score < _float(gates.get("min_entry_score"), 95):
        trend_gate_ok = False; gate_failures.append("score_gate")
    if htf_score < _float(gates.get("min_htf_score"), 70):
        trend_gate_ok = False; gate_failures.append("htf_gate")
    if adx < _float(gates.get("min_adx"), 22):
        trend_gate_ok = False; gate_failures.append("adx_gate")
    if rsi > _float(gates.get("max_rsi"), 70):
        trend_gate_ok = False; gate_failures.append("rsi_gate")
    if vol_ratio < _float(gates.get("min_vol_ratio"), 0.85):
        trend_gate_ok = False; gate_failures.append("vol_gate")
    if bool(gates.get("require_btc_ok", True)) and btc_trend_ok is not True:
        trend_gate_ok = False; gate_failures.append("btc_gate")

    exhausted_rsi = _float(ae_cfg.get("exhausted_rsi"), 72.0)
    if is_chop:
        trade_class = "CHOP_RISK"
    elif rsi >= exhausted_rsi and cont < 78:
        trade_class = "EXHAUSTED"
    elif cont >= _float(ae_cfg.get("trend_runner_min_score"), 78.0) and trend_gate_ok:
        trade_class = "TREND_RUNNER"
    elif cont <= _float(ae_cfg.get("scalp_exit_max_score"), 43.0):
        trade_class = "SCALP_EXIT"
    else:
        trade_class = "NORMAL"

    if gate_failures and cont >= _float(ae_cfg.get("trend_runner_min_score"), 78.0):
        reasons.append("runner_gate_blocked=" + ",".join(gate_failures))

    if bool(ae_cfg.get("strict_chop_guard", True)) and is_chop and trade_class == "TREND_RUNNER":
        trade_class = "CHOP_RISK"
        reasons.append("strict_chop_guard forced CHOP_RISK")

    if str(side).upper() == "SHORT" and trade_class == "TREND_RUNNER":
        trade_class = "NORMAL"
        reasons.append("short_runner_disabled")

    policy = _policy_from_cfg(trade_class, ae_cfg)
    confidence = max(0.0, min(100.0, abs(cont - 50.0) * 2.0))

    return AdaptiveExitDecision(
        enabled=True,
        shadow_mode=shadow,
        trade_class=trade_class,
        continuation_score=round(cont, 3),
        confidence=round(confidence, 3),
        policy=policy,
        reasons="; ".join(reasons),
        regime=regime_u,
        symbol_behavior_score=round(sym_score, 3),
        symbol_sample_weight=round(sym_weight, 3),
    )


def decision_note(decision: AdaptiveExitDecision) -> str:
    return (
        f"AE class={decision.trade_class} cont={decision.continuation_score:.1f} "
        f"conf={decision.confidence:.1f} policy={decision.policy.name} "
        f"shadow={int(decision.shadow_mode)} regime={decision.regime}"
    )
