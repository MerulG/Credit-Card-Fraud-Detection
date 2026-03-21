import os
import pandas as pd
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODELS_DIR = os.getenv("MODELS_DIR", "models")
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.856"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(MODELS_DIR)
    joblib_files = sorted(model_dir.glob("*.joblib"))
    if not joblib_files:
        raise RuntimeError(f"No .joblib files found in {MODELS_DIR}")
    latest = joblib_files[-1]
    state["pipeline"] = joblib.load(latest)
    state["model_file"] = latest.name
    state["threshold"] = FRAUD_THRESHOLD
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


class FraudFeatures(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Time": 406.0,
                    "V1": -1.3598071336738,
                    "V2": -0.0727811733098497,
                    "V3": 2.53634673796914,
                    "V4": 1.37815522427443,
                    "V5": -0.338320769942518,
                    "V6": 0.462387777762292,
                    "V7": 0.239598554061257,
                    "V8": 0.0986979012610507,
                    "V9": 0.363786969611213,
                    "V10": 0.0907941719789316,
                    "V11": -0.551599533260813,
                    "V12": -0.617800855762348,
                    "V13": -0.991389847235408,
                    "V14": -0.311169353699879,
                    "V15": 1.46817697209427,
                    "V16": -0.470400525259478,
                    "V17": 0.207971241929242,
                    "V18": 0.0257905801985591,
                    "V19": 0.403992960255733,
                    "V20": 0.251412098239705,
                    "V21": -0.018306777944153,
                    "V22": 0.277837575558899,
                    "V23": -0.110473910188767,
                    "V24": 0.0669280749146731,
                    "V25": 0.128539358273528,
                    "V26": -0.189114843888824,
                    "V27": 0.133558376740387,
                    "V28": -0.0210530534538215,
                    "Amount": 149.62
                }
            ]
        }
    }


FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def features_to_array(features: FraudFeatures) -> pd.DataFrame:
    return pd.DataFrame([[getattr(features, f) for f in FEATURE_ORDER]], columns=FEATURE_ORDER)


@app.get("/health")
def health():
    return {"status": "ok", "model": state["model_file"], "threshold": state["threshold"]}


@app.post("/predict")
def predict(features: FraudFeatures):
    try:
        arr = features_to_array(features)
        prob = float(state["pipeline"].predict_proba(arr)[:, 1][0])
        pred = 1 if prob >= state["threshold"] else 0
        return {"fraud_probability": prob, "prediction": pred, "threshold_used": state["threshold"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
def predict_batch(records: list[FraudFeatures]):
    if len(records) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 1000 records")
    try:
        arr = pd.concat([features_to_array(r) for r in records], ignore_index=True)
        probs = state["pipeline"].predict_proba(arr)[:, 1]
        threshold = state["threshold"]
        results = []
        for prob in probs:
            prob = float(prob)
            pred = 1 if prob >= threshold else 0
            results.append({"fraud_probability": prob, "prediction": pred, "threshold_used": threshold})
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
