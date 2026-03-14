from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os
import json

from src.predictor import HeartRiskPredictor

app = FastAPI(title="Heart Risk Prediction API")

predictor = HeartRiskPredictor()


@app.get("/")
def root():
    return {"message": "API is working"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        result = predictor.predict_from_csv(temp_path)

        safe_predictions = json.loads(result.to_json(orient="records"))

        return {
            "status": "success",
            "predictions": safe_predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.remove(temp_path)

