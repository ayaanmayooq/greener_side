import joblib, pandas as pd
from utils.config import DATA_ROOT, CFG
from research.features.swing.supervised.featset_prob_tp_v1 import build_feature_set_prob_tp_v1, FEATURE_COLUMNS_V1

MODEL_ID = "swing_xgb_prob_tp_v1"

if __name__ == "__main__":
    bundle = joblib.load(DATA_ROOT / f"research/models/swing/{MODEL_ID}/model.joblib")
    model, feats = bundle["model"], bundle["features"]

    for sym in CFG["symbols"]["swing"]:
        df = pd.read_parquet(DATA_ROOT / f"processed/daily/symbol={sym}/bars.parquet").sort_index()
        fe = build_feature_set_prob_tp_v1(df)
        X = fe[feats].astype("float32").dropna()
        p = pd.Series(model.predict_proba(X)[:,1], index=X.index, name="p_tp")
        outp = DATA_ROOT / f"research/preds/model={MODEL_ID}/freq=daily/symbol={sym}.parquet"
        outp.parent.mkdir(parents=True, exist_ok=True)
        p.to_frame().to_parquet(outp)
        print("wrote", outp, p.shape)