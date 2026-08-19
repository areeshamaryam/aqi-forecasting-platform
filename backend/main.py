"""
FastAPI backend for the Pearls AQI Predictor dashboard.

"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model_service import ModelService

app = FastAPI(
    title="Pearls AQI Predictor API",
    description=(
        "Serves 72-hour hourly AQI forecasts for Islamabad, "
        "backed by a multi-output Ridge Regression model "
        "registered in Hopsworks."
    ),
    version="1.0.0",
)

# --------------------------------------------------------
# CORS: allow the React frontend (running on a different
# port during development) to call this API from the
# browser.
# --------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend
                           # URL before deploying publicly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------
# The model + Hopsworks connection is loaded ONCE when the
# server starts, not on every request, since connecting to
# Hopsworks and downloading the model both take a few
# seconds.
# --------------------------------------------------------

model_service = ModelService()


@app.on_event("startup")
def startup_event():
    print("Starting up: connecting to Hopsworks and loading model...")
    model_service.load()
    print("✅ Startup complete. Model and feature connection ready.")


@app.get("/health")
def health():
    return {
        "status": "ok" if model_service.is_ready() else "not_ready",
        "model_version": model_service.model_version,
    }


@app.get("/predict")
def predict():
    """
    Returns the full 72-hour hourly AQI forecast for the
    configured city, along with a SHAP-based explanation of
    the 24-hour-ahead prediction and hazard alert flags.
    """

    if not model_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model is not ready yet. Try again shortly.",
        )

    try:
        result = model_service.predict()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )

    return result
