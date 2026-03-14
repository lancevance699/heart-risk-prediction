from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / 'models' / 'model_final.pkl'
THRESHOLD_PATH = BASE_DIR / 'models' / 'best_threshold.pkl'
FEATURES_PATH = BASE_DIR / 'models' / 'feature_columns.pkl'