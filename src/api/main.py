from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from src.api.inference_service import InferenceService


app = FastAPI(
    title="ArqueoVision API",
    description="Clasificación de estilos arquitectónicos a partir de imágenes",
    version="0.1.0",
)

# Ajusta esto luego con tu dominio real o tu puerto de frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = InferenceService()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": True,
        "classes": service.class_names,
    }


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Imagen no válida") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Error leyendo el archivo") from exc

    predictions = service.predict_pil(image=image, top_k=3)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "top_predictions": predictions,
        "prediction": predictions[0]["label"],
        "confidence": predictions[0]["score"],
    }