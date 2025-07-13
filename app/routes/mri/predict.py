from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
import random

router = APIRouter()

CLASSES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

@router.post("/mri/predict")
async def predict(file: UploadFile):
    prediction = random.choice(CLASSES)
    confidence = round(random.uniform(0.8, 0.99), 2)
    return JSONResponse(content={
        "prediction": prediction,
        "confidence": f"{confidence * 100}%"
    })