import pandas as pd
from research.features.swing.rules.core_indicators import swing_features

def build_feature_set_prob_tp_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features meant for classification of TP-before-SL inside horizon.
    Safe shifts; no lookahead.
    """
    fe = swing_features(df)  # mom_10, mom_20, sma_200, atr_14, ret_1
    fe["ret_5"]   = df["adj_close"].pct_change(5)
    fe["ret_20"]  = df["adj_close"].pct_change(20)
    fe["vol20"]   = fe["ret_1"].rolling(20).std() * (252 ** 0.5)
    fe["vol60"]   = fe["ret_1"].rolling(60).std() * (252 ** 0.5)
    fe["price_sma200_z"] = (df["adj_close"] - fe["sma_200"]) / (fe["atr_14"] + 1e-9)
    fe["vol_change20"]   = df["volume"].pct_change().rolling(20).mean()
    fe = fe.dropna()
    return fe

FEATURE_COLUMNS_V1 = [
    "mom_10","mom_20","ret_5","ret_20","atr_14","vol20","vol60","price_sma200_z","vol_change20"
]