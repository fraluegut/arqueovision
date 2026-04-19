import time
from datetime import datetime, timezone
from io import BytesIO

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.orm import Session

from src.api.database import Base, engine, get_db
from src.api.inference_service import InferenceService
from src.api.models import InferenceLog

# Create tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="ArqueoVision API",
    description="Clasificación de estilos arquitectónicos a partir de imágenes",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

service = InferenceService()
MODEL_NAME = "resnet18"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": service.model_loaded,
        "classes": service.class_names,
    }


@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    t_start = time.perf_counter()

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Imagen no válida") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Error leyendo el archivo") from exc

    try:
        predictions = service.predict_pil(image=image, top_k=3)
        latency_ms = (time.perf_counter() - t_start) * 1000

        log = InferenceLog(
            timestamp=datetime.now(timezone.utc),
            filename=file.filename or "",
            prediction=predictions[0]["label"],
            confidence=predictions[0]["score"],
            top_predictions=predictions,
            status="ok",
            model_name=MODEL_NAME,
            latency_ms=round(latency_ms, 2),
            error_detail=None,
        )
        db.add(log)
        db.commit()

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "top_predictions": predictions,
            "prediction": predictions[0]["label"],
            "confidence": predictions[0]["score"],
            "latency_ms": round(latency_ms, 2),
        }

    except Exception as exc:
        latency_ms = (time.perf_counter() - t_start) * 1000
        log = InferenceLog(
            timestamp=datetime.now(timezone.utc),
            filename=file.filename or "",
            prediction=None,
            confidence=None,
            top_predictions=None,
            status="error",
            model_name=MODEL_NAME,
            latency_ms=round(latency_ms, 2),
            error_detail=str(exc),
        )
        db.add(log)
        db.commit()
        raise HTTPException(status_code=500, detail="Error durante la inferencia") from exc


@app.get("/logs")
def get_logs(
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict:
    total = db.query(InferenceLog).count()
    rows = (
        db.query(InferenceLog)
        .order_by(InferenceLog.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "logs": [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "filename": r.filename,
                "prediction": r.prediction,
                "confidence": r.confidence,
                "top_predictions": r.top_predictions,
                "status": r.status,
                "model_name": r.model_name,
                "latency_ms": r.latency_ms,
                "error_detail": r.error_detail,
            }
            for r in rows
        ],
    }
