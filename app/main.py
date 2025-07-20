from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.mri import predict  # Import your router

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the MRI prediction route
app.include_router(predict.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Alzheimer's Prediction API. Go to /docs to try it."}