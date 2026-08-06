from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.parquet"

# -----------------------------
# Hopsworks
# -----------------------------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT= os.getenv("HOPSWORKS_PROJECT")

# -----------------------------
# Feature Group
# -----------------------------
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1