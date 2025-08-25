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

def simple_long_entries(
    df,
    *,
    mode: str = "roll",        # "flip" | "state" | "pullback" | "roll"
    adx_cut: float = 18.0,
    min_votes: int = 3,
    pullback_lookback: int = 3,
    roll_every: int = 10
):
    """
    Unified entry generator for your MonthlyTrend-like regime.

    mode:
      - "flip":     ON only on OFF→ON flip (breakout entry)
      - "state":    ON every bar while regime is ON (continuous exposure)
      - "pullback": ON when regime is ON AND we dipped below EMA5 in last N bars
                    AND we reclaim EMA5 today (buy-the-dip inside trend)
      - "roll":     ON at flip AND every `roll_every` bars thereafter while ON
    """
    df = df.copy()
    idx = pd.to_datetime(df.index)
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

    flip = long_state & (~long_state.shift(1, fill_value=False))

    if mode == "flip":
        entries = flip

    elif mode == "state":
        # every bar while ON
        entries = long_state.astype(bool)

    elif mode == "pullback":
        # had a dip below EMA5 recently, and today reclaimed above EMA5
        L = max(1, int(pullback_lookback))
        dipped  = (close < ema5).rolling(L).max().astype(bool)
        reclaim = (close > ema5) & (close.shift(1) <= ema5.shift(1))
        entries = (long_state & dipped & reclaim).astype(bool)

    elif mode == "roll":
        # fire at flip and then every `roll_every` bars inside the same ON segment
        # build segment ids (each ON run after a flip)
        seg_id = flip.cumsum().to_numpy()  # 0 while never flipped; 1,2,3... for each segment
        seg_pos = np.zeros(len(idx), dtype=int) - 1  # -1 = not in an ON segment yet
        seen = {}
        for i, sid in enumerate(seg_id):
            if sid == 0 or not long_state.iloc[i]:
                seg_pos[i] = -1
            else:
                prev = seen.get(sid, -1)
                seg_pos[i] = prev + 1
                seen[sid] = seg_pos[i]
        seg_pos = pd.Series(seg_pos, index=idx)

        rolls = (long_state) & (seg_pos > 0) & ((seg_pos % int(roll_every)) == 0)
        entries = (flip | rolls).astype(bool)

    else:
        raise ValueError("mode must be one of: 'flip', 'state', 'pullback', 'roll'")
    
    entries.index = idx
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


# ========= REAL-TIME ENTRY HELPERS =========

def entry_today(df: pd.DataFrame, adx_cut: float = 18.0, min_votes: int = 3):
    """
    True if the *latest* bar is a fresh long entry (state flipped OFF -> ON).
    Uses your simple_long_entries(df, adx_cut, min_votes).
    """
    if not {'Open','High','Low','Close'}.issubset(df.columns):
        raise ValueError("df must have Open/High/Low/Close")
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    entries = simple_long_entries(df, adx_cut=adx_cut, min_votes=min_votes)
    if len(entries) < 2:
        return False, None
    fired = bool(entries.iloc[-1])
    when  = df.index[-1] if fired else None
    return fired, when


def recommend_vertical_now(
    df: pd.DataFrame,
    *,
    N=20,
    exec_timing='next_open',     # 'next_open' (safer) or 'same_close' (faster)
    strike_inc=1.0,
    r=0.0, q=0.0,
    min_width_pct=0.03, max_width_pct=0.05,
    atr_mult=2.0,
    target_cost_max=0.40,
    force=False  # if True, ignore entry_today() and always return a rec
):
    """
    If today's bar fired a new long entry, build a live call-debit-vertical recommendation.
    - For 'same_close': uses today's Close as price anchor.
    - For 'next_open': uses today's Close as *planning* anchor; you will execute at next open.
      (So strikes may be a touch off in reality—good enough for a decision.)
    Returns a dict or None.
    """
    if not {'Open','High','Low','Close'}.issubset(df.columns):
        raise ValueError("df must have Open/High/Low/Close")
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if not force:
        fired, signal_ts = entry_today(df)
        if not fired:
            return None
    else:
        # pretend the latest bar fired
        signal_ts = df.index[-1]

    close = df['Close']
    # planning anchor for strikes/debit calc
    S_anchor = float(close.iloc[-1])

    # realized vol & ATR at the signal bar
    sigma = float(realized_vol_from_close(close, 20).bfill().iloc[-1] or 0.30)
    atr_v = float(atr14(df['High'], df['Low'], df['Close']).bfill().iloc[-1] or (S_anchor * 0.02))

    rec = suggest_call_vertical_for_entry(
        S_anchor, sigma, atr_v,
        N=N, strike_inc=strike_inc,
        min_width_pct=min_width_pct, max_width_pct=max_width_pct,
        atr_mult=atr_mult, target_cost_max=target_cost_max,
        r=r, q=q
    )

    # annotate execution intent + anchors
    rec['signal_date']  = signal_ts
    rec['exec_timing']  = exec_timing
    rec['price_anchor'] = {'field': 'Close', 'value': S_anchor}
    rec['comment']      = (
        "Exec at same_close uses Close[-1]. "
        "Exec at next_open will fill near tomorrow's open; strikes were planned off today's close."
    )
    return rec


def size_spread(rec: dict, equity: float, budget_frac: float = 0.10, multiplier: int = 100):
    """
    Convert a recommendation (with 'debit') into contracts and premium.
    - budget = equity * budget_frac
    - contracts = floor(budget / (debit * multiplier))
    Also reports planned risk ~= budget_frac * SL_on_prem (e.g., 0.10 * 0.30 = 3% of equity).
    """
    debit = float(rec['debit'])
    prem_per_spread = debit * float(multiplier)
    budget = float(equity) * float(budget_frac)

    contracts = int(budget // prem_per_spread) if prem_per_spread > 0 else 0
    premium_used = contracts * prem_per_spread

    sl_on_prem = float(rec.get('SL_on_prem', 0.50))
    planned_risk_pct_of_equity = budget_frac * sl_on_prem       # planned loss if SL hits
    worst_case_loss_pct        = budget_frac                     # max debit-at-risk if no exit

    return {
        "contracts": contracts,
        "premium_used": premium_used,
        "planned_risk_pct_of_equity": planned_risk_pct_of_equity,
        "worst_case_loss_pct_of_equity": worst_case_loss_pct,
        "notes": "planned risk assumes you can exit near SL; gaps/liquidity can be worse."
    }