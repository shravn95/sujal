import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Resolve .env relative to the backend directory (parent of app/)
_backend_dir = Path(__file__).resolve().parent.parent
_env_path = _backend_dir / ".env"

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Data Analyst API"
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_BASE_URL: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    
    # Support both model keys so all services find their expected variable
    NVIDIA_MODEL: str = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    NEMOTRON_MODEL: str = os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
    MAX_UPLOAD_SIZE_MB: int = 100

    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
    LLM_UNDERSTAND_TIMEOUT_SECONDS: float = 150.0   # reasoning pass
    LLM_PLAN_TIMEOUT_SECONDS: float = 90.0    # planning pass
    
    # Explicitly defined STORAGE_DIR required by workspace.py and data_cleaner.py
    STORAGE_DIR: str = os.getenv("STORAGE_DIR", "app/storage/datasets")

    @property
    def is_llm_configured(self) -> bool:
        """Check if the LLM API key is set and available."""
        return bool(self.NVIDIA_API_KEY and self.NVIDIA_API_KEY.strip())

    def __repr__(self) -> str:
        """Prevent accidental logging of the API key."""
        return (
            f"Settings(NVIDIA_BASE_URL={self.NVIDIA_BASE_URL!r}, "
            f"NVIDIA_MODEL={self.NVIDIA_MODEL!r}, "
            f"NVIDIA_API_KEY={'***' if self.NVIDIA_API_KEY else '(not set)'}, "
            f"STORAGE_DIR={self.STORAGE_DIR!r})"
        )

    __str__ = __repr__

    class Config:
        env_file = str(_env_path)
        extra = "ignore"

settings = Settings()

# Ensure storage directory exists on startup
os.makedirs(settings.STORAGE_DIR, exist_ok=True)