# research/features/registry.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

# base indicators for swing
from research.features.swing.rules.core_indicators import swing_features

def build_features(symbol: str,
                   bars: pd.DataFrame,
                   feature_keys: list[str],
                   data_root: Path) -> pd.DataFrame:
    """
    Build exactly the features a strategy asks for, by keys.

    Known keys:
      - "swing_core"               -> momentum/ATR/SMA200 on EOD bars
      - "prob_tp:<MODEL_ID>"       -> join calibrated p_tp predictions from disk

    Returns a DataFrame indexed like 'bars' (subset of dates where all requested features exist).
    Raises clear errors if a requested source is missing.
    """
    if not feature_keys:
        raise ValueError("feature_keys cannot be empty")

    fe: pd.DataFrame | None = None

    for key in feature_keys:
        if key == "swing_core":
            base = swing_features(bars)  # keeps OHLCV columns from bars + adds indicators
            fe = base if fe is None else fe.join(base, how="inner")

        elif key.startswith("prob_tp:"):
            model_id = key.split(":", 1)[1]
            pred_path = data_root / f"research/preds/model={model_id}/freq=daily/symbol={symbol}.parquet"
            if not pred_path.exists():
                raise FileNotFoundError(
                    f"Missing predictions for {symbol} at {pred_path}. "
                    f"Run inference for model '{model_id}' first."
                )
            p = pd.read_parquet(pred_path)["p_tp"]
            fe = (fe if fe is not None else swing_features(bars))
            fe = fe.join(p, how="left")
            fe = fe.dropna(subset=["p_tp"])

        else:
            raise ValueError(f"Unknown feature key: {key}")

    return fe.sort_index()