import numpy as np
import pandas as pd
from collections import deque

from options.suggest_options import bs_price_greeks

def vertical_price_given_strikes(S, T_years, sigma, K_long, K_short, r=0.0, q=0.0):
    longC  = bs_price_greeks(S, K_long,  T_years, r, q, sigma, 'call')
    shortC = bs_price_greeks(S, K_short, T_years, r, q, sigma, 'call')
    return max(1e-10, longC["price"] - shortC["price"])

def simulate_from_recs(
    df, recs,
    *,
    N=None,                     # if None, use rec['DTE']
    tp_on_prem=0.80,            # absolute TP (ROI on premium) if tp_mode in {'abs','cap'}
    tp_mode='cap',              # 'abs' | 'frac_of_max' | 'cap' (abs capped by tp_target*max)
    tp_target=0.85,             # used by 'frac_of_max' or as cap in 'cap'
    sl_on_prem=0.50,            # BASE SL = -50% on premium (before adaptives)
    exec_timing='next_open',    # 'next_open' (default) or 'same_close'
    r=0.0, q=0.0,
    # ---- adaptive exits ----
    be_after_roi=0.30,          # lock breakeven after +30% ROI on premium
    trail_after_roi=0.60,       # start trailing after +60% ROI
    trail_frac=0.40,            # giveback 40% of the peak premium after trail starts
    min_progress_midtime=0.10,  # by half DTE, require at least +10% ROI or exit
    ema_guard=False,            # if True, exit if Close < EMA(20)
    ema_span=20,
    # ---- sizing / reporting ----
    start_equity=100_000,
    budget_frac=None,           # e.g., 0.01 for 1% premium per trade
    multiplier=100,
    # ---- compounding / cash realism ----
    compound_on_close=False,    # if True, grow equity as trades close
    reserve_premium=False       # if True, deduct premium at entry and add it back at exit
):
    """
    Simulates recommended verticals (sticky sigma) with adaptive exits and sizing.
    Supports fixed-budget (non-compounding) or compounding-on-close with optional
    premium reservation.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if not {'Close','Open'}.issubset(df.columns):
        raise ValueError("df must have 'Open' and 'Close' columns")

    idx = df.index
    ema_series = df['Close'].ewm(span=ema_span, adjust=False).mean() if ema_guard else None

    need_cols = {'K_long','K_short','sigma'}
    missing = need_cols - set(recs.columns)
    if missing:
        raise ValueError(f"recs is missing columns {missing}. Run suggest_verticals_from_df first.")
    recs = recs.sort_index()

    rows = []
    eq = float(start_equity)
    pending = deque()  # queued cashflows for compounding: (exit_datetime, cashflow_dollars)

    for dt, rrow in recs.iterrows():
        dt = pd.to_datetime(dt)

        # locate signal day
        if dt in idx:
            i_sig = idx.get_loc(dt)
        else:
            i_sig = int(idx.searchsorted(dt))
            if i_sig >= len(idx):
                continue

        # execution timing
        if exec_timing == 'same_close':
            i_exec = i_sig
            if i_exec >= len(idx):
                continue
            S_exec = float(df['Close'].iloc[i_exec])
        else:  # 'next_open'
            i_exec = i_sig + 1
            if i_exec >= len(idx):
                continue
            S_exec = float(df['Open'].iloc[i_exec])

        N_trade = int(N if N is not None else rrow.get('DTE', 20))
        if N_trade <= 0:
            continue

        sigma0  = float(rrow.get('sigma', 0.30)) if np.isfinite(rrow.get('sigma', np.nan)) else 0.30
        K_long  = float(rrow['K_long'])
        K_short = float(rrow['K_short'])
        width   = float(K_short - K_long)

        # entry pricing
        T0 = N_trade / 252.0
        debit0 = vertical_price_given_strikes(S_exec, T0, sigma0, K_long, K_short, r=r, q=q)

        # max attainable ROI on premium for this spread
        roi_max = max((width - debit0) / max(debit0, 1e-12), 0.0)

        # choose TP
        mode = str(tp_mode).lower()
        if mode == 'abs':
            tp_roi = float(tp_on_prem)                    # e.g., 0.8 = +80%
        elif mode == 'frac_of_max':
            tp_roi = float(tp_target) * roi_max           # e.g., 0.85 * roi_max
        else:  # 'cap' -> abs TP capped by fraction of max
            tp_roi = min(float(tp_on_prem), float(tp_target) * roi_max)

        tp_level = debit0 * (1.0 + tp_roi)

        # base SL before adaptives kick in
        base_sl_level = debit0 * (1.0 - float(sl_on_prem))

        # simulate on CLOSE marks
        exit_idx = None
        value_at_exit = None
        reason = None

        peak_value   = debit0
        be_floor     = -np.inf   # breakeven floor
        trail_floor  = -np.inf   # trailing floor
        last_value   = debit0

        for j in range(1, N_trade + 1):
            k = i_exec + j
            if k >= len(idx):
                break

            Sj = float(df['Close'].iloc[k])
            Tj = max(1e-6, (N_trade - j) / 252.0)
            value = vertical_price_given_strikes(Sj, Tj, sigma0, K_long, K_short, r=r, q=q)
            last_value = value

            # progress & peak
            roi_now = (value - debit0) / max(debit0, 1e-12)
            peak_value = max(peak_value, value)
            elapsed_frac = j / max(N_trade, 1)

            # 1) breakeven lock
            if roi_now >= float(be_after_roi):
                be_floor = max(be_floor, debit0)

            # 2) trailing stop (giveback off the peak) after threshold
            if roi_now >= float(trail_after_roi):
                trail_floor = max(trail_floor, peak_value * (1.0 - float(trail_frac)))

            # 3) no-progress mid-time exit
            if (elapsed_frac >= 0.5) and (roi_now < float(min_progress_midtime)):
                exit_idx = k; value_at_exit = value; reason = 'NO_PROGRESS_50%'; break

            # 4) optional EMA guard
            if ema_guard and (float(df['Close'].iloc[k]) < float(ema_series.iloc[k])):
                exit_idx = k; value_at_exit = value; reason = 'EMA_EXIT'; break

            # assemble current SL
            curr_sl = max(base_sl_level, be_floor, trail_floor)

            # stop / TP checks
            if value >= tp_level:
                exit_idx = k; value_at_exit = value; reason = 'TP'; break
            if value <= curr_sl:
                exit_idx = k; value_at_exit = value; reason = 'ADAPTIVE_SL'; break

        # time stop
        if exit_idx is None:
            exit_idx = min(i_exec + N_trade, len(idx) - 1)
            value_at_exit = last_value
            reason = 'TIME'

        dt_exec = idx[i_exec]
        dt_exit = idx[exit_idx]
        ST      = float(df['Close'].iloc[exit_idx])

        roi = (value_at_exit - debit0) / max(debit0, 1e-12)

        # settle any exits that happened before this entry (for compounding)
        if compound_on_close and budget_frac:
            while pending and pending[0][0] <= dt_exec:
                _, cf = pending.popleft()
                eq += cf

        # sizing
        contracts = 0
        premium_used = 0.0
        pnl_dollars = 0.0
        if budget_frac is not None and budget_frac > 0:
            # budget source: fixed (start_equity) vs compounding (current eq)
            if compound_on_close:
                budget = eq * float(budget_frac)
            else:
                budget = float(start_equity) * float(budget_frac)

            prem_per_spread = debit0 * multiplier
            contracts = int(budget // prem_per_spread)
            premium_used = contracts * prem_per_spread

            # if reserving premium, remove it from equity at entry
            if compound_on_close and reserve_premium and contracts > 0:
                eq -= premium_used

            if contracts > 0:
                pnl_dollars = contracts * (value_at_exit - debit0) * multiplier
                # enqueue compounding cashflow at exit
                if compound_on_close:
                    cashflow = pnl_dollars + (premium_used if reserve_premium else 0.0)
                    pending.append((dt_exit, cashflow))

        rows.append({
            "signal_date": recs.index[recs.index.get_loc(dt)],
            "exec_date": dt_exec, "exit_date": dt_exit,
            "hold_days": (exit_idx - i_exec),
            "K_long": K_long, "K_short": K_short, "width": width,
            "S_exec": S_exec, "S_exit": ST,
            "debit0": debit0, "exit_value": value_at_exit,
            "roi": roi, "reason": reason,
            "contracts": contracts, "premium_used": premium_used, "pnl_dollars": pnl_dollars,
            "tp_roi": tp_roi, "roi_max": roi_max
        })

    trades = pd.DataFrame(rows).sort_values("exec_date")
    if trades.empty:
        return trades, {"trades": 0}

    # ---------- summary ----------
    win_mask  = trades['roi'] > 0
    loss_mask = ~win_mask

    win_rate       = float(win_mask.mean())
    avg_hold_days  = float(trades['hold_days'].mean())
    mean_roi       = float(trades['roi'].mean())
    median_roi     = float(trades['roi'].median())
    gross_mult     = float(np.prod(1.0 + trades['roi'].clip(lower=-1.0)))

    # ROI-based RR / PF / Expectancy
    avg_win_roi    = float(trades.loc[win_mask,  'roi'].mean()) if win_mask.any()  else 0.0
    avg_loss_roi   = float((-trades.loc[loss_mask,'roi']).mean()) if loss_mask.any() else 0.0  # magnitude
    rr_roi         = (avg_win_roi / avg_loss_roi) if avg_loss_roi > 0 else float('inf')
    pf_roi         = (float(trades.loc[win_mask,  'roi'].sum()) /
                      float((-trades.loc[loss_mask,'roi']).sum())) if loss_mask.any() else float('inf')
    expectancy_roi = win_rate * avg_win_roi - (1.0 - win_rate) * avg_loss_roi
    breakeven_winrate_roi = 1.0 / (1.0 + rr_roi) if np.isfinite(rr_roi) and rr_roi > 0 else float('nan')

    summary = {
        "trades": int(len(trades)),
        "wins": int(win_mask.sum()),
        "losses": int(loss_mask.sum()),
        "win_rate": win_rate,
        "mean_roi": mean_roi,
        "median_roi": median_roi,
        "gross_mult": gross_mult,
        "avg_hold_days": avg_hold_days,
        "exec_timing": exec_timing,
        "tp_on_prem": float(tp_on_prem),
        "tp_mode": tp_mode,
        "tp_target": float(tp_target),
        "sl_on_prem": float(sl_on_prem),
        "be_after_roi": float(be_after_roi),
        "trail_after_roi": float(trail_after_roi),
        "trail_frac": float(trail_frac),
        "min_progress_midtime": float(min_progress_midtime),
        "ema_guard": bool(ema_guard),
        # ROI-based risk metrics
        "avg_win_roi": avg_win_roi,
        "avg_loss_roi_mag": avg_loss_roi,
        "rr_roi": rr_roi,
        "profit_factor_roi": pf_roi,
        "expectancy_roi": expectancy_roi,
        "breakeven_winrate_roi": breakeven_winrate_roi,
        "winrate_x_rr": win_rate * rr_roi,
    }

    # Dollars section (only if you sized trades with budget_frac)
    if budget_frac:
        sum_pnl   = float(trades['pnl_dollars'].sum())
        sum_prem  = float(trades['premium_used'].sum())
        avg_win_d = float(trades.loc[win_mask,  'pnl_dollars'].mean())  if win_mask.any()  else 0.0
        avg_loss_d= float((-trades.loc[loss_mask,'pnl_dollars']).mean()) if loss_mask.any() else 0.0
        rr_d      = (avg_win_d / avg_loss_d) if avg_loss_d > 0 else float('inf')
        pf_d      = (float(trades.loc[win_mask,  'pnl_dollars'].sum()) /
                     float((-trades.loc[loss_mask,'pnl_dollars']).sum())) if loss_mask.any() else float('inf')
        expect_d  = win_rate * avg_win_d - (1.0 - win_rate) * avg_loss_d
        breakeven_winrate_d = 1.0 / (1.0 + rr_d) if np.isfinite(rr_d) and rr_d > 0 else float('nan')

        summary.update({
            "budget_frac": float(budget_frac),
            "start_equity": float(start_equity),
            "sum_pnl_dollars": sum_pnl,
            "sum_premium_used": sum_prem,
            "capital_efficiency": (sum_pnl / sum_prem) if sum_prem > 0 else float('nan'),
            "end_equity_fixed_budget": float(start_equity + sum_pnl),
            # $-based risk metrics
            "avg_win_dollars": avg_win_d,
            "avg_loss_dollars_mag": avg_loss_d,
            "rr_dollars": rr_d,
            "profit_factor_dollars": pf_d,
            "expectancy_dollars": expect_d,
            "breakeven_winrate_dollars": breakeven_winrate_d,
            "winrate_x_rr_dollars": win_rate * rr_d,
        })

        # finalize compounding equity (if enabled)
        if compound_on_close:
            # settle any remaining exits
            while pending:
                _, cf = pending.popleft()
                eq += cf
            summary["end_equity_compounded"] = float(eq)

    return trades, summary