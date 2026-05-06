import io
import os
import cv2
import base64
import traceback
import numpy as np

from PIL import Image
from datetime import datetime
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Depends,
    Header
)

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..utils.logger import setup_logger
from ..utils.config import Config
from ..inference.predictor import Predictor
from ..inference.grad_cam import GradCAMVisualizer

from .database import (
    init_db,
    add_api_key,
    validate_api_key,
    log_usage,
    get_usage
)

logger = setup_logger(__name__)

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
DEFAULT_API_KEY = os.getenv("DR_API_KEY", "dr_default_key")
APP_VERSION = "3.1.0"

predictor = None
gradcam = None


# ---------------------------------------------------
# LIFESPAN (RENDER SAFE)
# ---------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):

    global predictor, gradcam

    try:
        logger.info("🚀 Starting backend...")

        # Create directories
        Config.create_directories()

        # Init DB
        init_db()
        add_api_key(DEFAULT_API_KEY, "admin")

        # Model path
        model_path = Config.get_model_path()

        logger.info(f"📦 Loading model: {model_path}")

        predictor = Predictor(model_path=str(model_path))

        # GradCAM
        try:
            gradcam = GradCAMVisualizer(
                predictor.get_model()
            )
            logger.info("✅ GradCAM initialized")

        except Exception as e:
            gradcam = None
            logger.warning(f"⚠️ GradCAM disabled: {e}")

        app.state.predictor = predictor
        app.state.gradcam = gradcam

        logger.info("✅ Backend ready")

    except Exception as e:

        logger.error("❌ Startup failed")
        logger.error(str(e))

        traceback.print_exc()

        # IMPORTANT
        # do NOT crash Render deployment
        app.state.predictor = None
        app.state.gradcam = None

    yield

    logger.info("🛑 Backend shutdown")


# ---------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------
app = FastAPI(
    title="Retinal Disease Classification API",
    version=APP_VERSION,
    lifespan=lifespan
)

# ---------------------------------------------------
# CORS
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
# AUTH
# ---------------------------------------------------
def verify_api_key(
    x_api_key: Optional[str] = Header(None)
):

    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key missing"
        )

    if not validate_api_key(x_api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

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
# IMAGE LOADER
# ---------------------------------------------------
def load_image(contents):

    try:
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        return np.array(image)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )


# ---------------------------------------------------
# HEALTH
# ---------------------------------------------------
@app.get(
    "/health",
    response_model=HealthCheckResponse
)
async def health_check():

    status = (
        "healthy"
        if app.state.predictor is not None
        else "model_not_loaded"
    )

    return HealthCheckResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        version=APP_VERSION
    )


# ---------------------------------------------------
# USAGE
# ---------------------------------------------------
@app.get("/usage")
async def usage(
    api_key: str = Depends(verify_api_key)
):
    return get_usage(api_key)


# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):

    if app.state.predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    contents = await file.read()

    image_np = load_image(contents)

    try:

        label, confidence, probs = (
            app.state.predictor.predict(image_np)
        )

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
            gradcam_available=(
                app.state.gradcam is not None
            )
        )

    except Exception as e:

        logger.error(str(e))

        log_usage(api_key, False)

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )


# ---------------------------------------------------
# PREDICT WITH GRADCAM
# ---------------------------------------------------
@app.post("/predict-with-gradcam")
async def predict_with_gradcam(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):

    if app.state.predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    contents = await file.read()

    image_np = load_image(contents)

    try:

        label, confidence, probs = (
            app.state.predictor.predict(image_np)
        )

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

                processed = (
                    app.state.predictor.preprocess(
                        image_np
                    )
                )

                processed = np.expand_dims(
                    processed,
                    axis=0
                )

                class_idx = int(np.argmax(probs))

                heatmap = (
                    app.state.gradcam.generate_cam(
                        processed,
                        class_idx
                    )
                )

                overlay = (
                    app.state.gradcam.overlay_heatmap(
                        image_np,
                        heatmap
                    )
                )

                success, buffer = cv2.imencode(
                    ".png",
                    overlay
                )

                if success:

                    gradcam_b64 = (
                        base64.b64encode(
                            buffer.tobytes()
                        ).decode()
                    )

            except Exception as e:

                logger.warning(
                    f"GradCAM failed: {str(e)}"
                )

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

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )


# ---------------------------------------------------
# LOCAL RUN
# ---------------------------------------------------
if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port
    )