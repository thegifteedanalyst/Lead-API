from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("lead_model.pkl")
scaler = joblib.load("scaler.pkl")


class LeadData(BaseModel):
    sessions_count: int
    pages_viewed: int
    pricing_page_views: int
    time_on_site_sec: int
    recency_days: int
    deal_value: float


@app.post("/score")
def score_lead(data: LeadData):

    df = pd.DataFrame([data.dict()])

    X = scaler.transform(df)

    prob = model.predict_proba(X)[0][1]
    priority_score = prob * df["deal_value"].values[0]

    if priority_score >= 5000:
        priority = "HIGH"
    elif priority_score >= 2000:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    return {
        "conversion_probability": float(prob),
        "priority_score": float(priority_score),
        "priority_level": priority
    }