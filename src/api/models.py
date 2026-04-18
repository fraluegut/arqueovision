from datetime import datetime, timezone

from sqlalchemy import JSON, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.api.database import Base


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc), index=True
    )
    filename: Mapped[str] = mapped_column(String(512))
    prediction: Mapped[str | None] = mapped_column(String(256), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    top_predictions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16))        # "ok" | "error"
    model_name: Mapped[str] = mapped_column(String(256))
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
