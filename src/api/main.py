"""
FastAPI Application
RESTful API for retinal disease classification
"""

import os
import io
import base64
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
from typing import Dict
from datetime import datetime

from ..utils.logger import setup_logger
from ..utils.config import Config
from ..inference import Predictor, GradCAMVisualizer

logger = setup_logger(__name__)


# ---------------------------------------------------
# RESPONSE MODELS
# ---------------------------------------------------

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


# ---------------------------------------------------
# APP FACTORY
# ---------------------------------------------------

def create_app() -> FastAPI:

    app = FastAPI(
        title="Retinal Disease Classification API",
        description="AI-Based Retinal Disease Classification System",
        version="1.0.0"
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.predictor = None
    app.state.predictions_log = []

    # ---------------------------------------------------
    # STARTUP
    # ---------------------------------------------------

    @app.on_event("startup")
    async def startup_event():

        try:
            model_path = Config.MODEL_FULL_PATH

            if model_path and model_path.exists():
                app.state.predictor = Predictor(
                    str(model_path),
                    Config.DISEASE_CLASSES
                )
                logger.info(f"✅ Model loaded: {model_path}")
            else:
                logger.warning("❌ No model found in checkpoints")

        except Exception as e:
            logger.error(f"Startup error: {str(e)}")

    # ---------------------------------------------------
    # HEALTH CHECK
    # ---------------------------------------------------

    @app.get("/health-check", response_model=HealthCheckResponse)
    async def health_check():

        return HealthCheckResponse(
            status="healthy" if app.state.predictor else "model_not_loaded",
            timestamp=datetime.now().isoformat(),
            version="1.0.0"
        )

    # ---------------------------------------------------
    # MODEL INFO
    # ---------------------------------------------------

    @app.get("/model-info", response_model=ModelInfoResponse)
    async def model_info():

        if not app.state.predictor:
            raise HTTPException(status_code=503, detail="Model not loaded")

        model_file = os.path.basename(app.state.predictor.model_path)

        classes_dict = {
            str(k): v for k, v in Config.DISEASE_CLASSES.items()
        }

        return ModelInfoResponse(
            model_name=model_file,
            num_classes=Config.NUM_CLASSES,
            classes=classes_dict,
            input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3)
        )

    # ---------------------------------------------------
    # PREDICT
    # ---------------------------------------------------

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

            prob_dist = app.state.predictor.get_prediction_confidence_distribution(
                probabilities
            )

            app.state.predictions_log.append({
                "timestamp": datetime.now().isoformat(),
                "filename": file.filename,
                "prediction": predicted_class,
                "confidence": confidence
            })

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

    # ---------------------------------------------------
    # PREDICT WITH GRADCAM
    # ---------------------------------------------------

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

                    heatmap = visualizer.generate_cam(
                        np.expand_dims(image_array, axis=0),
                        class_idx
                    )

                    overlay = visualizer.overlay_heatmap(
                        image_array,
                        heatmap
                    )

                    # ✅ Pylance-safe fix
                    success, buffer = cv2.imencode(".png", overlay)  # type: ignore

                    if success:
                        gradcam_b64 = base64.b64encode(
                            buffer.tobytes()
                        ).decode()

            except Exception as e:
                logger.warning(f"GradCAM failed: {str(e)}")

            prob_dist = app.state.predictor.get_prediction_confidence_distribution(
                probabilities
            )

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

    # ---------------------------------------------------
    # PREDICTION LOG
    # ---------------------------------------------------

    @app.get("/predictions-log")
    async def get_predictions_log():

        return {
            "total_predictions": len(app.state.predictions_log),
            "predictions": app.state.predictions_log[-100:]
        }

    return app


# ---------------------------------------------------
# APP INSTANCE
# ---------------------------------------------------

app = create_app()


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.DEBUG
    )