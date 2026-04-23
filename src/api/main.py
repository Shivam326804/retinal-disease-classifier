import io
import base64
import numpy as np
import cv2
import os

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from typing import Dict, Optional
from datetime import datetime

from ..utils.logger import setup_logger
from ..utils.config import Config
from ..inference.predictor import Predictor
from ..inference.grad_cam import GradCAMVisualizer

# 🔥 NEW (DB)
from .database import (
    init_db,
    add_api_key,
    validate_api_key,
    log_usage,
    get_usage
)

logger = setup_logger(__name__)

# ---------------------------------------------------
# 🔐 API KEY (ENV SAFE)
# ---------------------------------------------------
DEFAULT_API_KEY = os.getenv("DR_API_KEY", "dr_ai_secure_key_123")

# ---------------------------------------------------
# 🔐 AUTH
# ---------------------------------------------------
def verify_api_key(x_api_key: Optional[str] = Header(None)):

    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key missing")

    if not validate_api_key(x_api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")

    return x_api_key


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


# ---------------------------------------------------
# APP FACTORY
# ---------------------------------------------------
def create_app() -> FastAPI:

    app = FastAPI(
        title="Retinal Disease Classification API",
        version="3.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.predictor = None
    app.state.gradcam = None

    # ---------------------------------------------------
    # STARTUP
    # ---------------------------------------------------
    @app.on_event("startup")
    async def startup_event():

        try:
            # 🔥 INIT DATABASE
            init_db()
            add_api_key(DEFAULT_API_KEY, "admin")

            model_path = Config.get_model_path()

            predictor = Predictor(model_path=str(model_path))

            gradcam = None
            try:
                gradcam = GradCAMVisualizer(predictor.get_model())
            except Exception as e:
                logger.warning(f"GradCAM disabled: {e}")

            app.state.predictor = predictor
            app.state.gradcam = gradcam

            logger.info("✅ Backend ready (DB + Model loaded)")

        except Exception as e:
            logger.error(f"Startup error: {str(e)}")

    # ---------------------------------------------------
    # LOGIN (BASIC)
    # ---------------------------------------------------
    @app.post("/login")
    async def login(username: str, password: str):

        if username == "admin" and password == "password123":
            return {
                "api_key": DEFAULT_API_KEY,
                "message": "Login successful"
            }

        raise HTTPException(status_code=401, detail="Invalid credentials")

    # ---------------------------------------------------
    # HEALTH
    # ---------------------------------------------------
    @app.get("/health-check", response_model=HealthCheckResponse)
    async def health_check():

        return HealthCheckResponse(
            status="healthy" if app.state.predictor else "model_not_loaded",
            timestamp=datetime.now().isoformat(),
            version="3.0.0"
        )

    # ---------------------------------------------------
    # USAGE (DB)
    # ---------------------------------------------------
    @app.get("/usage")
    async def usage(api_key: str = Depends(verify_api_key)):
        return get_usage(api_key)

    # ---------------------------------------------------
    # IMAGE LOADER
    # ---------------------------------------------------
    def load_image(contents):

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            return np.array(image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

    # ---------------------------------------------------
    # PREDICT
    # ---------------------------------------------------
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: UploadFile = File(...),
        api_key: str = Depends(verify_api_key)
    ):

        contents = await file.read()
        image_np = load_image(contents)

        try:
            label, confidence, probs = app.state.predictor.predict(image_np)

            prob_dict = {
                cls: float(p)
                for cls, p in zip(
                    app.state.predictor.get_classes(),
                    probs
                )
            }

            log_usage(api_key, True)

            return PredictionResponse(
                predicted_disease=label,
                confidence=float(confidence),
                probabilities=prob_dict,
                timestamp=datetime.now().isoformat(),
                gradcam_available=app.state.gradcam is not None
            )

        except Exception as e:
            logger.error(str(e))
            log_usage(api_key, False)
            raise HTTPException(status_code=500, detail="Prediction failed")

    # ---------------------------------------------------
    # PREDICT + GRADCAM
    # ---------------------------------------------------
    @app.post("/predict-with-gradcam")
    async def predict_with_gradcam(
        file: UploadFile = File(...),
        api_key: str = Depends(verify_api_key)
    ):

        contents = await file.read()
        image_np = load_image(contents)

        try:
            label, confidence, probs = app.state.predictor.predict(image_np)

            prob_dict = {
                cls: float(p)
                for cls, p in zip(
                    app.state.predictor.get_classes(),
                    probs
                )
            }

            gradcam_b64 = None

            if app.state.gradcam:
                try:
                    processed = app.state.predictor.preprocess(image_np)
                    processed = np.expand_dims(processed, axis=0)

                    class_idx = int(np.argmax(probs))

                    heatmap = app.state.gradcam.generate_cam(
                        processed, class_idx
                    )

                    overlay = app.state.gradcam.overlay_heatmap(
                        image_np, heatmap
                    )

                    success, buffer = cv2.imencode(".png", overlay)

                    if success:
                        gradcam_b64 = base64.b64encode(
                            buffer.tobytes()
                        ).decode()

                except Exception as e:
                    logger.warning(f"GradCAM failed: {str(e)}")

            log_usage(api_key, True)

            return JSONResponse(
                content={
                    "predicted_disease": label,
                    "confidence": float(confidence),
                    "probabilities": prob_dict,
                    "timestamp": datetime.now().isoformat(),
                    "gradcam_image": gradcam_b64
                }
            )

        except Exception as e:
            logger.error(str(e))
            log_usage(api_key, False)
            raise HTTPException(status_code=500, detail="Prediction failed")

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
        host="0.0.0.0",
        port=8000
    )