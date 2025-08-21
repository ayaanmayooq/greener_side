from typing import Tuple
import pandas as pd

def build_labels_triple_barrier(df: pd.DataFrame, G: float = 3.0, L: float = 1.0, h: int = 10,
                                atr_col: str = "atr_14", cost_bps: float = 2.0) -> pd.DataFrame:
    """
    Label t with y=1 if AFTER entering at t+1 OPEN, target (G*ATR) is hit before stop (L*ATR)
    within h days; else y=0. Uses daily OHLC to detect touches.
    Returns index=t with columns: y, p_be, entry_px, tp_px, sl_px, t_exit
    """
    out = []
    idx = df.index
    C = cost_bps / 10000.0
    p_be = (L + C) / (L + G)  # breakeven prob threshold given G/L and costs

    for i in range(len(df) - h - 2):
        entry_i = i + 1
        atr0 = df[atr_col].iloc[entry_i]
        if pd.isna(atr0):  # ATR not ready
            out.append((idx[i], None, p_be, None, None, None, None))
            continue
        entry_open = df["open"].iloc[entry_i]
        tp = entry_open + G * atr0
        sl = entry_open - L * atr0

        # walk forward up to horizon, detect first touch using highs/lows
        y, t_exit = 0, idx[min(entry_i + h, len(df) - 1)]
        for j in range(entry_i + 1, min(entry_i + h, len(df) - 1) + 1):
            if df["high"].iloc[j] >= tp:
                y, t_exit = 1, idx[j]; break
            if df["low"].iloc[j]  <= sl:
                y, t_exit = 0, idx[j]; break

        out.append((idx[i], y, p_be, entry_open, tp, sl, t_exit))

    lab = pd.DataFrame(out, columns=["t0","y","p_be","entry_px","tp_px","sl_px","t_exit"]).set_index("t0")
    return lab.dropna()

def build_labels_triple_barrier_touch_only(
    df: pd.DataFrame, G: float = 3.0, L: float = 1.0, h: int = 10,
    atr_col: str = "atr_14", cost_bps: float = 2.0
) -> pd.DataFrame:
    """
    Build labels using triple barrier but DROP 'NoTouch'.
    Returns rows where either TP or SL touched within horizon.
    Columns: y (1 if TP first else 0), p_be, entry_px, tp_px, sl_px, t_exit, status
    """
    out = []
    idx = df.index
    C = cost_bps / 10000.0
    p_be = (L + C) / (L + G)

    for i in range(len(df) - h - 2):
        entry_i = i + 1
        atr0 = df[atr_col].iloc[entry_i]
        if pd.isna(atr0): 
            continue
        entry_open = df["open"].iloc[entry_i]
        tp = entry_open + G * atr0
        sl = entry_open - L * atr0

        y, status, t_exit = 0, "NoTouch", idx[min(entry_i + h, len(df) - 1)]
        for j in range(entry_i + 1, min(entry_i + h, len(df) - 1) + 1):
            if df["high"].iloc[j] >= tp:
                y, status, t_exit = 1, "TP", idx[j]; break
            if df["low"].iloc[j]  <= sl:
                y, status, t_exit = 0, "SL", idx[j]; break

        if status != "NoTouch":
            out.append((idx[i], y, p_be, entry_open, tp, sl, t_exit, status))

    lab = pd.DataFrame(out, columns=["t0","y","p_be","entry_px","tp_px","sl_px","t_exit","status"]).set_index("t0")
    return lab