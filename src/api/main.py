"""
FastAPI Application
RESTful API for retinal disease classification
"""

import os
import io
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from typing import Dict
from datetime import datetime
from pathlib import Path

from ..utils.logger import setup_logger
from ..utils.config import Config
from ..inference import Predictor, GradCAMVisualizer

logger = setup_logger(__name__)


# ---------------- RESPONSE MODELS ----------------

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class PredictionResponse(BaseModel):
    predicted_disease: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str
    gradcam_available: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    num_classes: int
    classes: Dict[str, str]
    input_shape: tuple


# ---------------- APP FACTORY ----------------

def create_app() -> FastAPI:

    app = FastAPI(
        title="Retinal Disease Classification API",
        description="AI-Based Retinal Disease Classification System",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.predictor = None
    app.state.predictions_log = []

    # ---------------- STARTUP ----------------

    @app.on_event("startup")
    async def startup_event():

        try:

            primary_path = os.path.join(Config.MODELS_DIR, Config.MODEL_NAME)

            model_path = None

            if os.path.exists(primary_path):
                model_path = primary_path

            else:

                models_dir = Path(Config.MODELS_DIR)

                h5_files = list(models_dir.glob("*.h5"))

                if h5_files:
                    model_path = str(h5_files[0])
                    logger.warning(f"Using fallback model {model_path}")

            if model_path:
                app.state.predictor = Predictor(model_path, Config.DISEASE_CLASSES)
                logger.info(f"Predictor initialized using {model_path}")

            else:
                logger.warning("No model found")

        except Exception as e:
            logger.error(f"Startup error: {str(e)}")

    # ---------------- HEALTH CHECK ----------------

    @app.get("/health-check", response_model=HealthCheckResponse)
    async def health_check():

        return HealthCheckResponse(
            status="healthy" if app.state.predictor else "model_not_loaded",
            timestamp=datetime.now().isoformat(),
            version="1.0.0"
        )

    # ---------------- MODEL INFO ----------------

    @app.get("/model-info", response_model=ModelInfoResponse)
    async def model_info():

        if not app.state.predictor:
            raise HTTPException(status_code=503, detail="Model not loaded")

        model_file = os.path.basename(app.state.predictor.model_path)

        # convert int keys -> string keys
        classes_dict = {str(k): v for k, v in Config.DISEASE_CLASSES.items()}

        return ModelInfoResponse(
            model_name=model_file,
            num_classes=Config.NUM_CLASSES,
            classes=classes_dict,
            input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3)
        )

    # ---------------- PREDICT ----------------

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)):

        if not app.state.predictor:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:

            contents = await file.read()

            image = Image.open(io.BytesIO(contents)).convert("RGB")

            image = image.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))

            image_array = np.array(image).astype(np.float32) / 255.0

            predicted_class, confidence, probabilities = (
                app.state.predictor.predict(image_array)
            )

            prob_dist = app.state.predictor.get_prediction_confidence_distribution(probabilities)

            prediction_log = {
                "timestamp": datetime.now().isoformat(),
                "filename": file.filename,
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": prob_dist
            }

            app.state.predictions_log.append(prediction_log)

            return PredictionResponse(
                predicted_disease=predicted_class,
                confidence=float(confidence),
                probabilities=prob_dist,
                timestamp=datetime.now().isoformat(),
                gradcam_available=True
            )

        except Exception as e:
            logger.error(str(e))
            raise HTTPException(status_code=400, detail=str(e))

    # ---------------- PREDICT WITH GRADCAM ----------------

    @app.post("/predict-with-gradcam")
    async def predict_with_gradcam(file: UploadFile = File(...)):

        if not app.state.predictor:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:

            contents = await file.read()

            image = Image.open(io.BytesIO(contents)).convert("RGB")

            image = image.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))

            image_array = np.array(image).astype(np.float32) / 255.0

            predicted_class, confidence, probabilities = (
                app.state.predictor.predict(image_array)
            )

            gradcam_b64 = None

            try:

                if app.state.predictor.model:

                    visualizer = GradCAMVisualizer(app.state.predictor.model)

                    class_idx = int(np.argmax(probabilities))

                    heatmap = visualizer.generate_cam(image_array, class_idx)

                    visualization = visualizer.visualize_with_heatmap(image_array, heatmap)

                    success, buffer = cv2.imencode(".png", visualization)

                    if success:
                        gradcam_b64 = base64.b64encode(buffer.tobytes()).decode()

            except Exception as e:
                logger.warning(f"GradCAM failed: {str(e)}")

            prob_dist = app.state.predictor.get_prediction_confidence_distribution(probabilities)

            return JSONResponse(
                content={
                    "predicted_disease": predicted_class,
                    "confidence": float(confidence),
                    "probabilities": prob_dist,
                    "timestamp": datetime.now().isoformat(),
                    "gradcam_image": gradcam_b64
                }
            )

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ---------------- PREDICTION LOG ----------------

    @app.get("/predictions-log")
    async def get_predictions_log():

        return {
            "total_predictions": len(app.state.predictions_log),
            "predictions": app.state.predictions_log[-100:]
        }

    # ---------------- IMAGE UPLOAD ----------------

    @app.post("/upload-image")
    async def upload_image(file: UploadFile = File(...)):

        try:

            upload_dir = Path(Config.RAW_DATA_DIR) / "uploads"

            upload_dir.mkdir(parents=True, exist_ok=True)

            filename = file.filename if file.filename else f"upload_{datetime.now().timestamp()}.png"

            file_path = upload_dir / filename

            contents = await file.read()

            with open(file_path, "wb") as f:
                f.write(contents)

            return {
                "message": "Image uploaded successfully",
                "filename": filename,
                "path": str(file_path)
            }

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app


# ---------------- APP INSTANCE ----------------

app = create_app()


# ---------------- RUN SERVER ----------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        workers=Config.API_WORKERS,
        reload=Config.DEBUG
    )