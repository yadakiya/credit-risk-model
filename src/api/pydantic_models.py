from pydantic import BaseModel


class PredictionRequest(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int


class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int
