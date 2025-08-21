import json, joblib, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
from utils.config import DATA_ROOT, CFG
from research.features.swing.supervised.featset_prob_tp_v1 import build_feature_set_prob_tp_v1, FEATURE_COLUMNS_V1
from research.labels.swing.triple_barrier import build_labels_triple_barrier
from research.labels.swing.triple_barrier import build_labels_triple_barrier_touch_only as build_labels


MODEL_ID = "swing_xgb_prob_tp_v1"

def _normalize_bars(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Ensure single-level columns: open, high, low, close, volume, adj_close."""
    out = df.copy()

    # If columns are MultiIndex, try to slice the ticker level first
    if isinstance(out.columns, pd.MultiIndex):
        names = list(out.columns.names or [])
        if "Ticker" in names:
            # Try exact symbol, then lower-case (yfinance often lowercases)
            try:
                out = out.xs(sym, axis=1, level="Ticker")
            except KeyError:
                out = out.xs(sym.lower(), axis=1, level="Ticker")
        # If still multiindex, keep the first level (e.g., "Price")
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)

    # Normalize names
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    if "adjclose" in out.columns and "adj_close" not in out.columns:
        out = out.rename(columns={"adjclose": "adj_close"})
    return out

def _load_symbol(sym: str):
    p = DATA_ROOT / f"processed/daily/symbol={sym}/bars.parquet"
    raw = pd.read_parquet(p).sort_index()
    df  = _normalize_bars(raw, sym)

    # Build features (these ALREADY include open/high/low)
    fe = build_feature_set_prob_tp_v1(df)

    # touch-only labels, aligned with the SAME (G,L,h) you’ll use in backtest
    lab = build_labels(fe, G=3.0, L=1.0, h=10, atr_col="atr_14")

    X = fe.loc[lab.index, FEATURE_COLUMNS_V1].astype("float32").dropna()
    y = lab.loc[X.index, "y"].astype(int)
    return X, y

if __name__ == "__main__":
    Xs, ys = [], []
    for sym in CFG["symbols"]["swing"]:
        X, y = _load_symbol(sym); Xs.append(X); ys.append(y)
    X = pd.concat(Xs); y = pd.concat(ys)

    # simple time split
    cut = int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]

    base = XGBClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=4
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xtr, ytr)

    p = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p))
    brier = float(brier_score_loss(yte, p))
    print({"AUC": round(auc,4), "Brier": round(brier,4)})

    mdir = DATA_ROOT / "research/models/swing/swing_xgb_prob_tp_v1"
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": FEATURE_COLUMNS_V1}, mdir / "model.joblib")
    with open(mdir / "card.json","w") as f:
        json.dump({"model_id": MODEL_ID, "features": FEATURE_COLUMNS_V1, "auc":auc, "brier":brier}, f, indent=2)
    print("Saved", mdir)