import math
import numpy as np
import pandas as pd

# ---------- robust BS pricer ----------
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)
def _phi(x): return math.exp(-0.5*x*x) / _SQRT2PI
def _Phi(x): return 0.5 * (1.0 + math.erf(x / _SQRT2))

def _pos(x, eps=1e-12):
    """Return a safe positive float for things that must be >0."""
    try:
        xv = float(x)
    except Exception:
        return eps
    if not np.isfinite(xv) or xv <= 0:
        return eps
    return xv

def bs_price_greeks(S, K, T, r=0.0, q=0.0, sigma=0.30, kind='call'):
    """
    Black–Scholes-Merton, robust against NaNs/zeros.
    Returns dict(price, delta, gamma, theta, vega).
    """
    S    = _pos(S)
    K    = _pos(K)
    T    = max(1e-6, float(T))
    r    = float(r)
    q    = float(q)
    sig  = float(sigma)
    if not np.isfinite(sig) or sig <= 0:
        sig = 0.30  # sane default
    volT = max(1e-6, sig * math.sqrt(T))

    # safe forward ratio for log
    fwd   = S * math.exp((r - q) * T)
    fk    = max(fwd / K, 1e-15)
    d1    = (math.log(fk) / volT) + 0.5 * volT
    d2    = d1 - volT

    if kind == 'call':
        price = math.exp(-r*T) * (fwd * _Phi(d1) - K * _Phi(d2))
        delta = math.exp(-q*T) * _Phi(d1)
    else:
        price = math.exp(-r*T) * (K * _Phi(-d2) - fwd * _Phi(-d1))
        delta = math.exp(-q*T) * (_Phi(d1) - 1.0)

    gamma = math.exp(-q*T) * _phi(d1) / (S * volT)
    vega  = math.exp(-q*T) * S * _phi(d1) * math.sqrt(T)
    theta = -(S * math.exp(-q*T) * _phi(d1) * sig) / (2.0 * math.sqrt(T)) \
            - r * K * math.exp(-r*T) * (_Phi(d2) if kind=='call' else _Phi(-d2)) \
            + q * S * math.exp(-q*T) * (_Phi(d1) if kind=='call' else _Phi(-d1))

    return {"price": price, "delta": delta, "gamma": gamma, "theta": theta, "vega": vega}

def round_to_increment(x, inc=1.0):
    inc = float(inc) if np.isfinite(inc) and inc > 0 else 1.0
    return round(float(x) / inc) * inc

def build_call_vertical(S, T_years, sigma, *, width_abs, r=0.0, q=0.0, strike_inc=1.0):
    """
    ATM-ish call debit vertical. Guarantees K_high > K_low by >= 1 increment.
    """
    S = _pos(S)
    width_abs = float(width_abs)
    inc = float(strike_inc) if np.isfinite(strike_inc) and strike_inc > 0 else 1.0

    # ensure width at least one tick
    width_abs = max(width_abs, inc)

    K_low  = round_to_increment(S, inc)
    K_high = round_to_increment(S + width_abs, inc)
    if K_high <= K_low:
        K_high = K_low + inc  # nudge up one increment

    longC  = bs_price_greeks(S, K_low,  T_years, r, q, sigma, 'call')
    shortC = bs_price_greeks(S, K_high, T_years, r, q, sigma, 'call')
    debit  = max(1e-10, longC["price"] - shortC["price"])  # keep positive & finite
    net_delta = longC["delta"] - shortC["delta"]
    width = K_high - K_low
    return {
        "K_low":K_low, "K_high":K_high, "width":width,
        "debit":debit, "cost_frac": (debit / max(width, 1e-9)),
        "delta": net_delta
    }


def realized_vol_from_close(close, lookback=20):
    return close.pct_change().rolling(lookback).std() * np.sqrt(252.0)

def atr14(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def _rma(x: pd.Series, period: int):
    # Wilder's smoothing (RMA)
    return x.ewm(alpha=1/period, adjust=False).mean()

def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high, low, close = high.astype(float), low.astype(float), close.astype(float)
    up   = high.diff()
    down = -low.diff()

    plus_dm  = ((up > down) & (up > 0)).astype(float) * up
    minus_dm = ((down > up) & (down > 0)).astype(float) * down

    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr     = _rma(tr, period)
    plus_di = 100 * _rma(plus_dm, period)  / atr.replace(0, np.nan)
    minus_di= 100 * _rma(minus_dm, period) / atr.replace(0, np.nan)
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _rma(dx, period)

def simple_long_entries(df, adx_cut: float = 18.0, min_votes: int = 3):
    close = df['Close'].astype(float)
    high  = df['High'].astype(float)
    low   = df['Low'].astype(float)

    # ema3  = close.ewm(span=3,  adjust=False).mean()
    # ema10 = close.ewm(span=10, adjust=False).mean()

    ema5  = close.ewm(span=5,  adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    slope = (ema20 - ema20.shift(5)) > 0
    mom20 = (close / close.shift(20) - 1.0) > 0
    adx   = adx_wilder(high, low, close, period=14)
    # adx_cut = 12

    votes = (
        (ema5 > ema20).astype(int)
        + slope.astype(int)
        + mom20.astype(int)
        + (adx > adx_cut).astype(int)
    )
    long_state = (votes >= int(min_votes))
    entries = long_state & (~long_state.shift(1).fillna(False))
    return entries.astype(bool)

def suggest_call_vertical_for_entry(S, sigma, atr_val, *, 
                                    N=20, strike_inc=1.0,
                                    min_width_pct=0.03, max_width_pct=0.05,
                                    atr_mult=2.0, target_cost_max=0.40,
                                    r=0.0, q=0.0):
    T = N/252.0
    # width candidate: max(%, ATR multiple) but cap at max_width_pct*Spot
    width0 = max(min_width_pct*S, atr_mult*atr_val)
    width0 = min(width0, max_width_pct*S)

    for w in [width0, 0.04*S, 0.05*S]:
        v = build_call_vertical(S, T, sigma, width_abs=w, r=r, q=q, strike_inc=strike_inc)
        if v["cost_frac"] <= target_cost_max:
            debit = v["debit"]
            return {
                "kind":"call_debit_vertical","DTE":N,
                "S":S,"sigma":sigma,
                "K_long":v["K_low"],"K_short":v["K_high"],
                "width":v["width"],"debit":debit,
                "cost_frac":v["cost_frac"],"net_delta":v["delta"],
                "TP_on_prem":0.80,"SL_on_prem":0.50,
                "TP_level":debit*1.80,"SL_level":debit*0.50,
                "notes":f"width chosen to keep cost<= {target_cost_max:.0%}"
            }
    # all attempts were rich → still return, but warn
    debit = v["debit"]
    return {
        "kind":"call_debit_vertical","DTE":N,
        "S":S,"sigma":sigma,
        "K_long":v["K_low"],"K_short":v["K_high"],
        "width":v["width"],"debit":debit,
        "cost_frac":v["cost_frac"],"net_delta":v["delta"],
        "TP_on_prem":0.80,"SL_on_prem":0.50,
        "TP_level":debit*1.80,"SL_level":debit*0.50,
        "notes":"Spread came in rich; consider wider DTE/width or skip."
    }

def suggest_verticals_from_df(df, N=20, strike_inc=1.0):
    close = df['Close'].dropna()
    # deprecation-safe backfills
    sigma_series = realized_vol_from_close(close, 20).bfill().fillna(0.30)
    atr_series   = atr14(df['High'], df['Low'], df['Close']).bfill().fillna(close*0.02)

    entries = simple_long_entries(df)
    
    out = []
    for dt in close.index[entries]:
        S   = float(close.loc[dt])
        sig = float(sigma_series.loc[dt])
        a   = float(atr_series.loc[dt])

        # skip any bar with junk inputs
        if not (np.isfinite(S) and S > 0 and np.isfinite(sig) and sig > 0 and np.isfinite(a) and a >= 0):
            continue

        rec = suggest_call_vertical_for_entry(S, sig, a, N=N, strike_inc=strike_inc)
        rec['date'] = pd.to_datetime(dt)
        out.append(rec)

    if not out:
        return pd.DataFrame(columns=[
            'date','kind','DTE','S','sigma','K_long','K_short','width',
            'debit','cost_frac','net_delta','TP_on_prem','SL_on_prem','TP_level','SL_level','notes'
        ]).set_index('date')

    return pd.DataFrame(out).set_index('date').sort_index()