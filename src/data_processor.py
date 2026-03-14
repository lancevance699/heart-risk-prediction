import pandas as pd
import joblib


class DataProcessor:
    def __init__(self, features_path):
        self.feature_columns = joblib.load(features_path)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])

        if 'id' in data.columns:
            data = data.drop(columns=['id'])

        if 'Gender' in data.columns:
            data['Gender'] = data['Gender'].replace({
                '1.0': 'Male',
                '0.0': 'Female'
            })

        data = data.fillna(data.median(numeric_only=True))

        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = None

        data = data[self.feature_columns]

        return data