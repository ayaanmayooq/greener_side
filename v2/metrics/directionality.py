import numpy as np
import pandas as pd

# --- utilities ---------------------------------------------------------------

def _ensure_daily_index(x):
    """Return a daily DatetimeIndex (normalized to midnight), no duplicates."""
    idx = pd.DatetimeIndex(x)
    idx = idx.tz_localize(None) if idx.tz is not None else idx
    idx = idx.normalize()
    return pd.DatetimeIndex(pd.Series(index=idx, dtype=float).groupby(level=0).ngroup().index)

def _ann_ir(returns, fwdN):
    """Annualized IR over an N-bar horizon; safe against 0/NaN std."""
    s = pd.Series(returns).dropna()
    if s.size == 0: 
        return np.nan
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return np.nan
    return (s.mean() / std) * np.sqrt(252.0 / float(fwdN))

# --- signals -----------------------------------------------------------------

def monthly_ls_signal(close: pd.Series) -> pd.Series:
    """
    Very simple long/short/flat regime:
      +1 if EMA(5) > EMA(20) and EMA(20) rising over 5 bars,
      -1 if EMA(5) < EMA(20) and EMA(20) falling over 5 bars,
       0 otherwise.
    """
    c = close.copy()
    c.index = _ensure_daily_index(c.index)
    ema5  = c.ewm(span=5,  adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    slope = ema20 - ema20.shift(5)

    sig = pd.Series(0, index=c.index)
    sig[(ema5 > ema20) & (slope > 0)]  = 1
    sig[(ema5 < ema20) & (slope < 0)]  = -1
    return sig

def sma_longonly_signal(close: pd.Series) -> pd.Series:
    """Long-only: +1 when fast > slow, else 0."""
    c = close.copy()
    c.index = _ensure_daily_index(c.index)
    fast = c.ewm(span=10, adjust=False).mean()
    slow = c.ewm(span=30, adjust=False).mean()
    return (fast > slow).astype(int)

# --- diagnostics -------------------------------------------------------------

def directionality_from_signal(close: pd.Series, signal: pd.Series, fwdN: int = 20):
    """
    Computes forward returns and basic stats for a provided signal.
    Returns: (long_mask, short_mask, close_aligned)
    """
    # align & forward return
    c = close.copy()
    c.index = _ensure_daily_index(c.index)
    sig = signal.reindex(c.index).fillna(0)

    fwd = (c.shift(-fwdN) / c) - 1.0

    long_m  = sig == 1
    short_m = sig == -1
    flat_m  = sig == 0

    # hit rates (guard empties)
    long_hit  = float((fwd[long_m]  > 0).mean()) if long_m.any() else np.nan
    short_hit = float((fwd[short_m] < 0).mean()) if short_m.any() else np.nan

    # means & lift (guard empties)
    long_mean  = float(fwd[long_m].mean())  if long_m.any() else np.nan
    short_mean = float(fwd[short_m].mean()) if short_m.any() else np.nan
    flat_mean  = float(fwd[flat_m].mean())  if flat_m.any() else np.nan

    long_lift  = (long_mean - flat_mean) if np.isfinite(long_mean) and np.isfinite(flat_mean) else np.nan
    short_lift = ((-short_mean) - (-flat_mean)) if np.isfinite(short_mean) and np.isfinite(flat_mean) else np.nan

    # correlation (use signed signal, drop flats to avoid dilution)
    ic = fwd.corr(sig.replace(0, np.nan).dropna())

    ir_long  = _ann_ir(fwd[long_m],  fwdN) if long_m.any()  else np.nan
    ir_short = _ann_ir(-fwd[short_m], fwdN) if short_m.any() else np.nan

    exp_long  = float(long_m.mean())
    exp_short = float(short_m.mean())

    # pretty print without NaN spam
    def _fmt_pct(x):  return f"{x:.2%}" if np.isfinite(x) else "n/a"
    def _fmt_float(x):return f"{x:.4f}" if np.isfinite(x) else "n/a"
    def _fmt_ic(x):   return f"{x:.3f}" if np.isfinite(x) else "n/a"

    print(f"\n=== Directionality diagnostics (fwd {fwdN} bars) ===")
    print(f"Long hit:   {_fmt_pct(long_hit)} | mean fwd: {_fmt_float(long_mean)} | lift vs flat: {_fmt_float(long_lift)}")
    if (sig == -1).any():
        print(f"Short hit:  {_fmt_pct(short_hit)} | mean fwd: {_fmt_float(short_mean)} | (abs) lift: {_fmt_float(short_lift)}")
    print(f"IC (signal ↔ fwd): {_fmt_ic(ic)}")
    print(f"IR long: {_fmt_float(ir_long)}" + (f" | IR short: {_fmt_float(ir_short)}" if (sig==-1).any() else ""))
    print(f"Exposure: long {exp_long:.2%} | short {exp_short:.2%}")

    return long_m.astype(int), short_m.astype(int), c

# --- options overlay ---------------------------------------------------------

def estimate_options_overlay(close: pd.Series,
                             long_mask: pd.Series,
                             short_mask: pd.Series,
                             N: int = 20,
                             width_pct: float = 0.05,
                             cost_frac: float = 0.50) -> pd.DataFrame:
    """
    Naive vertical-spread sleeve:
      - On a long entry: buy ATM call spread with width = S0*width_pct, pay = cost_frac*width
      - On a short entry: buy ATM put spread with same economics
      - Exit after N bars; payoff clipped to spread width.
    Returns a DataFrame of per-trade results. Prints summary stats.
    """
    c = close.copy()
    c.index = _ensure_daily_index(c.index)
    long_m  = pd.Series(long_mask,  index=c.index).fillna(0).astype(int)
    short_m = pd.Series(short_mask, index=c.index).fillna(0).astype(int)

    state = (long_m - short_m)  # +1 long, -1 short, 0 flat
    entries = state.ne(state.shift(1).fillna(0)) & (state != 0)

    rows = []
    idx = c.index

    for dt in idx[entries]:
        i = idx.get_loc(dt)
        if i + N >= len(idx):
            break
        side = 'long_call_spread' if state.loc[dt] == 1 else 'long_put_spread'
        S0, ST = float(c.iloc[i]), float(c.iloc[i+N])
        width  = S0 * width_pct
        prem   = cost_frac * width

        if side == 'long_call_spread':
            payoff = float(np.clip(ST - S0, 0.0, width))
        else:
            payoff = float(np.clip(S0 - ST, 0.0, width))

        roi = (payoff - prem) / prem  # cap losses at -100% is handled in compounding step
        rows.append({
            'entry': dt, 'exit': idx[i+N], 'side': side,
            'S0': S0, 'ST': ST, 'width': width, 'premium': prem,
            'payoff': payoff, 'roi': roi
        })

    if not rows:
        print("\nOptions overlay: no trades detected (signal never changed state).")
        return pd.DataFrame(columns=['entry','exit','side','S0','ST','width','premium','payoff','roi'])

    opt = pd.DataFrame(rows).sort_values('entry').reset_index(drop=True)

    print(f"\n=== Options overlay (~{N} DTE, width {width_pct:.1%}, premium {cost_frac:.0%} of width) ===")
    print(f"Trades: {len(opt)} | Mean ROI: {opt.roi.mean():.2f} | Median ROI: {opt.roi.median():.2f} | Win%: {(opt.roi>0).mean():.2%}")
    gross_mult = float(np.prod(1.0 + opt['roi'].clip(lower=-1.0)))  # cap at -100%
    print(f"Naive compounded multiplier (per-trade sleeve): {gross_mult:.2f}x")

    return opt

def simple_directionality_check(df_bt, fwdN=20, mode='sma_long'):
    """No PyFolio. Recompute signal from prices and test if it points the next ~month."""
    close = df_bt['Close'].copy()
    close.index = pd.to_datetime(close.index)
    
    if mode == 'sma_long':
        ema_f = close.ewm(span=10, adjust=False).mean()
        ema_s = close.ewm(span=30, adjust=False).mean()
        signal = (ema_f > ema_s).astype(int)           # 1=long, 0=flat
    elif mode == 'monthly_ls':
        # Monthly Trend (L/S) very simply: +1 if 5>20 and 20 rising, -1 if 5<20 and 20 falling
        ema5  = close.ewm(span=5, adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        slope = (ema20 - ema20.shift(5))               # 1-week slope
        long_cond  = (ema5 > ema20) & (slope > 0)
        short_cond = (ema5 < ema20) & (slope < 0)
        signal = pd.Series(0, index=close.index)
        signal[long_cond]  = 1
        signal[short_cond] = -1
    else:
        raise ValueError("mode must be 'sma_long' or 'monthly_ls'")

    # Forward N‑day underlying return
    fwd = (close.shift(-fwdN) / close) - 1.0

    long_mask  = signal == 1
    short_mask = signal == -1
    flat_mask  = signal == 0

    # Hit rates
    long_hit  = (fwd[long_mask]  > 0).mean() if long_mask.any() else np.nan
    short_hit = (fwd[short_mask] < 0).mean() if short_mask.any() else np.nan

    # Means / lift
    long_mean  = fwd[long_mask].mean()  if long_mask.any() else np.nan
    short_mean = fwd[short_mask].mean() if short_mask.any() else np.nan
    flat_mean  = fwd[flat_mask].mean()  if flat_mask.any() else np.nan
    long_lift  = (long_mean - flat_mean) if np.isfinite(long_mean) and np.isfinite(flat_mean) else np.nan
    short_lift = ((-short_mean) - (-flat_mean)) if np.isfinite(short_mean) and np.isfinite(flat_mean) else np.nan

    # Simple “IC”: corr between signed signal and forward return
    ic = fwd.corr(signal.replace(0, np.nan).dropna())  # drop flats for stability
    # Conditional IR (annualized to 252d, over fwdN horizon)
    def ann_ir(x):
        x = x.dropna()
        return (x.mean()/x.std()) * np.sqrt(252.0 / fwdN) if x.std() not in [0, np.nan] else np.nan
    ir_long  = ann_ir(fwd[long_mask])   if long_mask.any() else np.nan
    ir_short = ann_ir(-fwd[short_mask]) if short_mask.any() else np.nan

    print(f"\n=== Simple directionality ({mode}, fwd {fwdN} bars) ===")
    print(f"Long hit:   {long_hit:.2%} | mean fwd: {long_mean:.4f} | lift vs flat: {long_lift:.4f}")
    if (signal == -1).any():
        print(f"Short hit:  {short_hit:.2%} | mean fwd: {short_mean:.4f} | (abs) lift:  {short_lift:.4f}")
    print(f"IC (signal ↔ fwd): {ic:.3f}")
    print(f"IR long: {ir_long:.2f}" + (f" | IR short: {ir_short:.2f}" if (signal==-1).any() else ""))
    print(f"Exposure: long {(long_mask.mean() if long_mask.any() else 0):.2%} | short {(short_mask.mean() if short_mask.any() else 0):.2%}")
    
    return signal, fwd
