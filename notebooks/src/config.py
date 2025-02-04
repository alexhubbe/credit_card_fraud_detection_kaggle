from pathlib import Path

PALETTE = 'cividis'
RANDOM_STATE = 42
SCATTER_ALPHA = 0.2


PROJECT_FOLDER = Path(__file__).resolve().parents[2]

DATA_FOLDER = PROJECT_FOLDER / "data"

# put the path for the project data files below
ORIGINAL_DATA = DATA_FOLDER / "creditcard.csv.zip"
CLEAN_DATA = DATA_FOLDER / "creditcard.parquet"

# put the path for the project model files below
MODELS_FOLDER = PROJECT_FOLDER / "models"
DEPLOYED_MODEL = MODELS_FOLDER/"best_model.pkl"

# put any other necessary paths below
REPORT_FOLDER = PROJECT_FOLDER / "reports"
IMAGES_FOLDER = REPORT_FOLDER / "images"

PREPROCESSOR_FIG = IMAGES_FOLDER/'preprocessor.jpg' 
HP_OPTIMIZATION_FIG = IMAGES_FOLDER/'hp_optimization.jpg'