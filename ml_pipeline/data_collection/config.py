import os
from pathlib import Path
from dotenv import load_dotenv

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# OpenWeather API Configuration
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not OPENWEATHER_API_KEY:
    raise ValueError(
        "OPENWEATHER_API_KEY not found. Please add it to your .env file."
    )

# API URLs
GEOCODING_API_URL = "http://api.openweathermap.org/geo/1.0/direct"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
AIR_POLLUTION_API_URL = "https://api.openweathermap.org/data/2.5/air_pollution"

# Request Configuration
REQUEST_TIMEOUT = 10

# Default City (for testing)
DEFAULT_CITY = "Islamabad"