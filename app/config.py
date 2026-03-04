from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    model_path: Path = Path("models/classifier.pt")
    max_slices: int = 200
    use_gpu: bool = False
    temp_dir: Path = Path("/tmp/dicom-demo")  # noqa: S108
    temp_retention_seconds: int = 3600
    log_level: str = "INFO"
    max_upload_size_mb: int = 500
    inference_batch_size: int = 8
    annotation_dir: Path = Path("data")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
