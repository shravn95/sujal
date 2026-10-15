from fastapi import APIRouter, Body
from app.agents.forecast_agent import forecasting_agent

router = APIRouter()

@router.post("/forecast")
async def generate_forecast(payload: dict = Body(...)):
    """
    Generates time-series predictive modeling with best, expected, and worst case scenario bounds.
    """
    historical = payload.get("historical_values", [4000, 4200, 4500, 4800, 5200, 5600])
    periods = payload.get("periods", 4)
    result = forecasting_agent.forecast_metric(historical, periods)
    return result
