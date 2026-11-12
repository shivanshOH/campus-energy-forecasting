"""Application configuration with safe local defaults."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings.

    Paths are resolved relative to the process working directory, which keeps
    CLI, API and container execution predictable.
    """

    app_env: str = "development"
    database_url: str = "sqlite:///./artifacts/campus_forecast.db"
    data_root: Path = Path("data")
    artifact_root: Path = Path("artifacts")
    max_upload_bytes: int = 500 * 1024 * 1024
    default_timezone: str = "Asia/Kolkata"
    default_interval: str = "10min"
    random_seed: int = 42

    model_config = SettingsConfigDict(
        env_prefix="CAMPUS_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def reports_dir(self) -> Path:
        return self.artifact_root / "reports"

    @property
    def checkpoints_dir(self) -> Path:
        return self.artifact_root / "checkpoints"

    def ensure_directories(self) -> None:
        for path in (
            self.data_root,
            self.artifact_root,
            self.reports_dir,
            self.checkpoints_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
