from fastapi import FastAPI

app = FastAPI(
    title="AutoML API",
    description="Automated Machine Learning Platform",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to AutoML API"
    }