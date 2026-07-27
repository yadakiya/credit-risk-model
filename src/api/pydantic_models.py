"""Request/response schemas for the Credit Risk API.

Field names here must match ``config.PipelineConfig.feature_columns``
exactly, since the model is trained on those column names in that order.
"""
from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    total_amount: float = Field(..., description="Sum of all transaction amounts for the customer")
    avg_amount: float = Field(..., description="Average transaction amount for the customer")
    transaction_count: int = Field(..., ge=0, description="Number of transactions for the customer")
    std_amount: float = Field(
        0.0, description="Std. deviation of the customer's transaction amounts"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_amount": 15000.0,
                "avg_amount": 750.0,
                "transaction_count": 20,
                "std_amount": 320.5,
            }
        }
    )


class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int
