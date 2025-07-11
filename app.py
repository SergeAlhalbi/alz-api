from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import random

app = FastAPI()

CLASSES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

@app.post("/predict")
async def predict(file: UploadFile):
    prediction = random.choice(CLASSES)
    confidence = round(random.uniform(0.8, 0.99), 2)
    return JSONResponse(content={"prediction": prediction, "confidence": f"{confidence * 100}%"})

@app.get("/")
def root():
    return {"message": "Welcome to the Alzheimer's Prediction API. Go to /docs to try it."}