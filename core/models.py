from typing import List

from pydantic import BaseModel


class Alert(BaseModel):
    timestamp: str
    score: float


class AlertsSummary(BaseModel):
    events: List[Alert]
    event_count: int
    max_score: float


