import math
import numpy as np
import pandas as pd

# --- Normal CDF/PDF without SciPy ---
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

def _phi(x):   # standard normal pdf
    return math.exp(-0.5*x*x) / SQRT2PI

def _Phi(x):   # standard normal cdf via erf
    return 0.5 * (1.0 + math.erf(x / SQRT2))

def bs_price_greeks(S, K, T, r=0.0, q=0.0, sigma=0.25, kind='call'):
    """
    Black–Scholes–Merton (European). Returns dict(price, delta, gamma, theta, vega).
    S: spot, K: strike, T: years to expiry, r: risk-free, q: div yield, sigma: vol
    kind: 'call' or 'put'
    """
    S = float(S); K = float(K); T = max(1e-6, float(T))
    r = float(r); q = float(q); sigma = max(1e-6, float(sigma))
    fwd = S * math.exp((r - q)*T)
    volT = sigma * math.sqrt(T)
    d1 = (math.log(fwd / K) / volT) + 0.5 * volT
    d2 = d1 - volT

    if kind == 'call':
        price = math.exp(-r*T) * (fwd * _Phi(d1) - K * _Phi(d2))
        delta = math.exp(-q*T) * _Phi(d1)
    else:
        price = math.exp(-r*T) * (K * _Phi(-d2) - fwd * _Phi(-d1))
        delta = math.exp(-q*T) * (_Phi(d1) - 1.0)

    gamma = math.exp(-q*T) * _phi(d1) / (S * volT)
    vega  = math.exp(-q*T) * S * _phi(d1) * math.sqrt(T)  # per 1 vol (i.e., 1.00 = 100 vols)
    # Theta (per year, Black–Scholes convention)
    theta = - (S * math.exp(-q*T) * _phi(d1) * sigma) / (2.0 * math.sqrt(T)) \
            - r * K * math.exp(-r*T) * ( _Phi(d2) if kind=='call' else _Phi(-d2) ) \
            + q * S * math.exp(-q*T) * ( _Phi(d1) if kind=='call' else _Phi(-d1) )

    return {"price": price, "delta": delta, "gamma": gamma, "theta": theta, "vega": vega}

def round_to_increment(x, inc=1.0):
    return round(x / inc) * inc

def build_vertical(S, T, r, q, sigma, width_pct=0.03, kind='call', strike_inc=1.0):
    """
    ATM vertical (long lower, short upper) with width as % of spot.
    Returns dict with legs and aggregate price/greeks.
    """
    width = S * float(width_pct)
    K_low  = round_to_increment(S, strike_inc)
    K_high = round_to_increment(S + width, strike_inc) if kind=='call' else round_to_increment(S - width, strike_inc)

    if kind == 'call':
        long_leg = bs_price_greeks(S, K_low,  T, r, q, sigma, 'call')
        short_leg= bs_price_greeks(S, K_high, T, r, q, sigma, 'call')
        delta = long_leg['delta'] - short_leg['delta']
    else:
        K_high = round_to_increment(S, strike_inc)           # higher K
        K_low  = round_to_increment(S - width, strike_inc)   # lower K
        long_leg = bs_price_greeks(S, K_high, T, r, q, sigma, 'put')
        short_leg= bs_price_greeks(S, K_low,  T, r, q, sigma, 'put')
        delta = -(long_leg['delta'] - short_leg['delta'])    # sign normalize (positive when benefits down)

    price = long_leg['price'] - short_leg['price']  # debit
    gamma = long_leg['gamma'] - short_leg['gamma']
    theta = long_leg['theta'] - short_leg['theta']
    vega  = long_leg['vega']  - short_leg['vega']
    width_abs = abs(K_high - K_low)
    max_payoff = width_abs  # per $1 underlying tick == $1 intrinsic at expiry (in underlying units)

    return {
        "kind": kind,
        "K_low": K_low, "K_high": K_high, "width": width_abs,
        "debit": price, "delta": delta, "gamma": gamma, "theta": theta, "vega": vega,
    }

def screen_vertical(v, max_cost_frac=0.35, target_delta_range=(0.35, 0.70), min_width=0.02, max_width=0.06, width_pct=None, S=None):
    """
    Heuristics to accept/reject a vertical:
    - debit ≤ max_cost_frac * width
    - net delta within target range (long call spread ~ 0.1–0.4 typically; we allow wider)
    - width between [min_width, max_width] of spot if S/width_pct supplied
    """
    ok = True
    reasons = []

    width = v["width"]
    cost_frac = v["debit"] / max(width, 1e-9)
    if cost_frac > max_cost_frac:
        ok = False; reasons.append(f"cost_frac {cost_frac:.2f} > {max_cost_frac:.2f}")

    net_delta = v["delta"]
    lo, hi = target_delta_range
    if not (lo <= abs(net_delta) <= hi):
        ok = False; reasons.append(f"|delta| {abs(net_delta):.2f} ∉ [{lo:.2f},{hi:.2f}]")

    if width_pct is not None and S is not None:
        w_pct = width / S
        if not (min_width <= w_pct <= max_width):
            ok = False; reasons.append(f"width_pct {w_pct:.3f} ∉ [{min_width:.3f},{max_width:.3f}]")

    return ok, reasons, {"cost_frac": cost_frac, "net_delta": net_delta}

def realized_vol(returns, lookback=20):
    return returns.rolling(lookback).std() * np.sqrt(252.0)

def backtest_vertical_overlay(
    close: pd.Series,
    long_entries: pd.Series,
    long_exits: pd.Series,
    *,
    N=20,                    # target holding days ~ DTE
    width_pct=0.03,
    r=0.00, q=0.00,
    vol_mode="rv20_sticky",  # 'rv20_sticky' or float like 0.25
    tp_on_premium=0.80,      # take profit when gain = +80% of premium
    sl_on_premium=0.50,      # stop loss when loss = -50% of premium
    screen_kwargs=None       # dict for screen_vertical (max_cost_frac, delta range...)
):
    """
    Returns: DataFrame with one row per trade + summary dict.
    """
    screen_kwargs = screen_kwargs or {}
    close = close.dropna().copy()
    ret = close.pct_change()
    rv20 = realized_vol(ret, 20)

    idx = close.index
    long_entries = pd.Series(long_entries, index=idx).fillna(False).astype(bool)
    long_exits   = pd.Series(long_exits,   index=idx).fillna(False).astype(bool)

    rows = []
    i = 0
    while i < len(idx):
        dt = idx[i]
        if long_entries.iloc[i]:
            # Entry state
            S0 = float(close.iloc[i])
            T  = N / 252.0
            sigma = float(rv20.iloc[i]) if vol_mode == "rv20_sticky" else float(vol_mode)
            # Build/screen spread
            v = build_vertical(S0, T, r, q, sigma, width_pct=width_pct, kind='call', strike_inc=1.0)
            ok, reasons, extras = screen_vertical(v, width_pct=width_pct, S=S0, **screen_kwargs)
            if not ok:
                rows.append({"entry": dt, "reason": "screen_fail", "notes": "; ".join(reasons)})
                i += 1
                continue

            debit0 = v["debit"]
            width  = v["width"]
            max_payoff = width
            tp_level = debit0 * (1.0 + float(tp_on_premium))
            sl_level = debit0 * (1.0 - float(sl_on_premium))

            # Simulate forward
            exit_idx = None
            value_at_exit = None
            reason = None
            for j in range(1, N+1):
                if i + j >= len(idx):
                    break
                dtj = idx[i + j]
                Sj  = float(close.iloc[i + j])
                Tj  = max(1e-6, (N - j) / 252.0)  # time decays
                # sticky sigma
                vj = build_vertical(Sj, Tj, r, q, sigma, width_pct=width_pct, kind='call', strike_inc=1.0)
                value = vj["debit"]

                # optional strategy exit: if your long_exits flips true, bail
                if long_exits.iloc[i + j]:
                    exit_idx = i + j
                    value_at_exit = value
                    reason = "signal_exit"
                    break

                if value >= tp_level:
                    exit_idx = i + j
                    value_at_exit = value
                    reason = "take_profit"
                    break
                if value <= sl_level:
                    exit_idx = i + j
                    value_at_exit = value
                    reason = "stop_loss"
                    break

            # time stop if still in
            if exit_idx is None:
                exit_idx = min(i + N, len(idx)-1)
                value_at_exit = value if 'value' in locals() else debit0
                reason = "time_stop"

            dt_exit = idx[exit_idx]
            roi = (value_at_exit - debit0) / max(debit0, 1e-9)

            rows.append({
                "entry": dt, "exit": dt_exit, "hold_days": (exit_idx - i),
                "S0": S0, "ST": float(close.iloc[exit_idx]),
                "width": width, "debit0": debit0, "value_exit": value_at_exit,
                "roi": roi, "reason": reason,
                "cost_frac": extras.get("cost_frac", np.nan), "delta": extras.get("net_delta", np.nan)
            })

            # advance: avoid overlap by jumping to exit+1
            i = exit_idx + 1
            continue

        i += 1

    trades = pd.DataFrame(rows)
    if trades.empty:
        summary = {"trades": 0}
        return trades, summary

    summary = {
        "trades": len(trades),
        "win_rate": float((trades["roi"] > 0).mean()),
        "mean_roi": float(trades["roi"].mean()),
        "median_roi": float(trades["roi"].median()),
        "gross_mult": float(np.prod(1.0 + trades["roi"].clip(lower=-1.0))),
        "avg_hold_days": float(trades["hold_days"].mean()),
        "mean_cost_frac": float(trades["cost_frac"].mean()),
    }
    return trades, summary