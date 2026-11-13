from fastapi import FastAPI
from backend.app.routes.dataset_routes import router as dataset_router



app = FastAPI(
    title="AutoML API",
    description="Automated Machine Learning Platform",
    version="1.0.0"
)

app.include_router(
    dataset_router,
    prefix="/api/datasets",
    tags=["Dataset Management"]
)

@app.get("/")
def home():
    return {
        "message": "Welcome to AutoML API"
    }