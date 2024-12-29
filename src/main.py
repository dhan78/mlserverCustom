from fastapi import FastAPI
from src.api.routes import router
from src.config.settings import get_settings
from src.utils.logging import setup_logging

logger = setup_logging()
settings = get_settings()

app = FastAPI(
    title="ML Server",
    description="Enterprise ML Server for XGBoost models",
    version="1.0.0"
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=True
    )