import pandas as pd
import joblib

from src.config import MODEL_PATH, THRESHOLD_PATH, FEATURES_PATH
from src.data_processor import DataProcessor


class HeartRiskPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.threshold = joblib.load(THRESHOLD_PATH)
        self.processor = DataProcessor(FEATURES_PATH)

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        ids = df['id'].tolist() if 'id' in df.columns else list(range(len(df)))

        X = self.processor.preprocess(df)
        proba = self.model.predict_proba(X)[:, 1]
        preds = (proba >= self.threshold).astype(int)

        result = pd.DataFrame({
            'id': ids,
            'prediction': preds
        })

        result['id'] = result['id'].astype(int)
        result['prediction'] = result['prediction'].astype(int)

        return result

    def predict_from_csv(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        return self.predict_from_dataframe(df)