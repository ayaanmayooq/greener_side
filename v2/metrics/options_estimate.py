def estimate_options_from_trade_analyzer(
    *,
    # —— from Backtrader / your script ——
    trades, won=None, winpct=None, avgwin=0.0, avgloss=0.0,
    start_equity=100_000,

    # —— sizing (what produced your avgwin/avgloss vs. what you want to model) ——
    backtest_sizer=0.10,          # was bt_sizer_pct (e.g., PercentSizer 10% -> 0.10)
    estimate_sizer=None,          # was est_sizer_pct (None -> backtest_sizer)

    # —— vertical spread economics ——
    spread_width_pct=0.03,        # was width_pct (2–5% typical)
    premium_cost_frac=0.30,       # was cost_frac  (25–50% typical)
    realism_beta=0.50,            # was beta (0.45–0.55 sane)
    win_capture_frac=0.60,        # was win_capture (0.55–0.70)

    # —— loss handling ——
    loser_stop_frac=0.50,         # NEW: cut losers at 50% premium loss (0.0–1.0)

    # —— how you deploy capital (pick one or compare all) ——
    per_trade_budget_frac=0.01,   # was premium_frac    (spend this % of START equity each trade)
    pot_budget_frac=0.01,         # was sleeve_frac     (one-time pot that compounds)
    equity_reinvest_frac=0.01,    # was reinvest_frac   (% of CURRENT equity each trade)

    verbose=True
):
    """
    Rough options estimator (single call).
    - Converts your underlying avg win/loss into a % move per “slice” of capital.
    - Maps that into vertical-spread ROI on premium (cheap toy model).
    - Applies your capital deployment style(s) to show end equity.

    Key ideas:
      * backtest_sizer = the position % used to PRODUCE avgwin/avgloss (e.g., 0.10)
      * estimate_sizer = the position % you want to MODEL now (e.g., 0.01)
      * loser_stop_frac = hard stop on losers as % of premium (0.50 = lose half)
    """
    # ---------- inputs ----------
    N = int(trades)
    if N <= 0:
        if verbose: print("[Options] No trades -> no estimate.")
        return None

    if estimate_sizer is None:
        estimate_sizer = backtest_sizer

    # Win rate: prefer explicit %; else compute from counts
    if winpct is not None:
        p_win = max(0.0, min(1.0, float(winpct) / 100.0))
    elif won is not None:
        p_win = max(0.0, min(1.0, float(won) / float(trades)))
    else:
        raise ValueError("Provide either winpct (percent) or won (count).")

    # Clamp economics
    w = max(float(spread_width_pct), 1e-6)
    c = min(max(float(premium_cost_frac), 1e-6), 0.99)
    beta = max(0.0, min(1.0, float(realism_beta)))
    win_cap = max(0.0, min(1.0, float(win_capture_frac)))
    stop_L = max(0.0, min(1.0, float(loser_stop_frac)))

    # ---------- rebase avg $ to the estimator sizer ----------
    scale = (float(estimate_sizer) / float(backtest_sizer)) if backtest_sizer > 0 else 1.0
    adj_avg_win  = float(avgwin)  * scale
    adj_avg_loss = float(avgloss) * scale  # (likely negative; we use abs below)

    # “Slice” of capital per trade under the estimator sizer
    slice_dollars = float(start_equity) * float(estimate_sizer)
    if slice_dollars <= 0:
        raise ValueError("estimate_sizer and start_equity must be > 0.")

    # Underlying % moves of that slice (sizer-invariant)
    win_pct_move  = adj_avg_win / slice_dollars
    loss_pct_move = abs(adj_avg_loss) / slice_dollars

    # ---------- map to vertical-spread ROI on premium ----------
    # progress toward cap (haircut + cap at 1)
    prog_win  = min(beta * (win_pct_move  / w), 1.0)
    # losers: you said "cut at 50% loss of premium" -> fixed ROI_loss = -stop_L
    # (If you want dynamic losses, replace next line with: prog_loss = min(beta*(loss_pct_move/w),1.0); roi_loss=-prog_loss)
    roi_max   = 1.0 / c - 1.0
    roi_win   = win_cap * prog_win * roi_max
    roi_loss  = -stop_L

    ev_per_trade = p_win * roi_win + (1.0 - p_win) * roi_loss

    # ---------- turn ROI-on-premium into end equity ----------
    # 1) Fixed $ budget each trade (no compounding of that budget)
    prem_fixed = start_equity * float(per_trade_budget_frac)
    equity_end_fixed = start_equity + prem_fixed * ev_per_trade * N

    # 2) One-time pot that compounds separately, then is added back
    pot_start = start_equity * float(pot_budget_frac)
    pot_end   = pot_start * (1.0 + ev_per_trade) ** N
    equity_end_pot = start_equity - pot_start + pot_end

    # 3) Reinvest a fraction of CURRENT equity each trade (portfolio-level compounding)
    equity_end_reinvest = start_equity * (1.0 + float(equity_reinvest_frac) * ev_per_trade) ** N

    if verbose:
        print("\n=== Rough Options Estimate ===")
        print(f"Trades={N}, win={p_win:.1%}, start=${start_equity:,.0f}")
        print(f"Sizers: backtest={backtest_sizer:.2%} -> estimate={estimate_sizer:.2%} (scale {scale:.2f})")
        print(f"Econ: width={w:.1%}, cost={c:.0%}, beta={beta:.2f}, win_cap={win_cap:.2f}, stop_loss={stop_L:.2f}")
        print(f"Adj avg win=${adj_avg_win:.2f}, adj avg loss=${adj_avg_loss:.2f}, slice=${slice_dollars:,.0f}")
        print(f"Win move={win_pct_move:.2%} -> progress={prog_win:.2f}")
        print(f"ROI_win≈{roi_win:.2f}, ROI_loss≈{roi_loss:.2f}  =>  EV/trade≈{ev_per_trade:.2%}")
        print(f"- End equity (fixed {per_trade_budget_frac:.3f}/trade): ${equity_end_fixed:,.2f}")
        print(f"- End equity (pot {pot_budget_frac:.3f} compounding): ${equity_end_pot:,.2f}")
        print(f"- End equity (reinvest {equity_reinvest_frac:.3f} of equity): ${equity_end_reinvest:,.2f}")

    return {
        "p_win": p_win, "trades": N,
        "adj_avg_win": adj_avg_win, "adj_avg_loss": adj_avg_loss,
        "slice_dollars": slice_dollars,
        "win_pct_move": win_pct_move, "loss_pct_move": loss_pct_move,
        "roi_win": roi_win, "roi_loss": roi_loss,
        "ev_per_trade": ev_per_trade,
        "equity_end": {
            "fixed_per_trade": equity_end_fixed,
            "compounding_pot": equity_end_pot,
            "reinvest_fraction": equity_end_reinvest
        }
    }