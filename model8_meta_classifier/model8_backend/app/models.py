from pydantic import BaseModel
from typing import Optional, List

class MarketStatus(BaseModel):
    tradable: bool
    volatility_regime: str
    macro_regime: str
    market_state: str
    confidence_threshold: float

class SignalResponse(BaseModel):
    issued: bool
    direction: Optional[str]
    confidence: Optional[float]
    expected_directional_accuracy: Optional[float]
    coverage_at_threshold: Optional[float]
    reason: Optional[str]

class PerformancePoint(BaseModel):
    threshold: float
    coverage: float
    directional_accuracy: float

class PerformanceMetrics(BaseModel):
    samples: int
    active_coverage: float
    directional_accuracy: float
    by_threshold: List[PerformancePoint]
