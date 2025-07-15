from fastapi import FastAPI
from app.routes.mri import predict  # Import your router

app = FastAPI()

# Include the MRI prediction route
app.include_router(predict.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Alzheimer's Prediction API. Go to /docs to try it."}