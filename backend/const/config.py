import os


class Config:
    """Application configuration from environment variables"""
    
    # Application Settings
    APP_NAME: str = "DiagWiki"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = os.environ.get("NODE_ENV", "development")
    PORT: int = int(os.environ.get("PORT", 8001))
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "Qwen3-Coder-30B-A3B-Instruct")
    
    @classmethod
    def is_development(cls) -> bool:
        return cls.ENVIRONMENT != "production"


# Export commonly used values
APP_NAME = Config.APP_NAME
APP_VERSION = Config.APP_VERSION
