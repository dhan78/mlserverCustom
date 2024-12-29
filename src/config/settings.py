from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings managed through environment variables."""
    
    # Model settings
    MODEL_PATH: str = "models/model.json"
    MODEL_VERSION: str = "v1"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Performance
    BATCH_SIZE: int = 32
    ENABLE_BATCHING: bool = True
    
    # Security
    API_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()