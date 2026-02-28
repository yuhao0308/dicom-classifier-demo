from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    model_path: Path = Field(default=Path("models/classifier.pt"), validation_alias="MODEL_PATH")
    max_slices: int = Field(default=100, validation_alias="MAX_SLICES")
    use_gpu: bool = Field(default=False, validation_alias="USE_GPU")
    temp_dir: Path = Field(default=Path("/tmp/dicom-demo"), validation_alias="TEMP_DIR")
    temp_retention_seconds: int = Field(default=3600, validation_alias="TEMP_RETENTION_SECONDS")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    max_upload_size_mb: int = Field(default=500, validation_alias="MAX_UPLOAD_SIZE_MB")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
