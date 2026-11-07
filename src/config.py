"""Project-wide constants. Run every command from the repository root."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "zameen.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# MLflow: a local SQLite database (works on every MLflow version, no server needed)
MLFLOW_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "lahore-house-prices"

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Zameen.com area units (Zameen defines 1 marla = 225 sq ft, 1 kanal = 20 marla)
MARLA_SQFT = 225.0
KANAL_SQFT = 20 * MARLA_SQFT

# Rule-based cleaning bounds (fixed rules, NOT percentiles, so the test set stays honest)
BOUNDS = {
    "price": (1_000_000, 3_000_000_000),   # PKR 10 lakh .. 300 crore
    "area_sqft": (450, 45_000),            # 2 marla .. 200 marla
    "bedrooms": (1, 15),
    "baths": (0, 15),
    "price_per_sqft": (500, 150_000),      # removes obvious typos / unit mistakes
}

# Rough reference points for distance features (approximate coordinates)
LAHORE_CENTER = (31.5497, 74.3436)
LIBERTY_MARKET = (31.5100, 74.3440)   # Gulberg commercial hub
LAHORE_BBOX = (31.20, 31.80, 74.00, 74.70)  # lat_min, lat_max, lon_min, lon_max
