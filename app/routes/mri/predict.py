from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from scripts.mri.inference import predict

router = APIRouter()
CLASSES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

@router.post("/mri/predict")
async def mri_predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predicted_class, probs = predict(image_bytes)

    return JSONResponse(content={
        "prediction": CLASSES[predicted_class],
        "confidence": f"{probs[predicted_class] * 100:.2f}%",
        "probabilities": {
            CLASSES[i]: f"{p * 100:.2f}%" for i, p in enumerate(probs)
        }
    })