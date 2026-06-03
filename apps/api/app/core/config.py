from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# repo root, so artifact paths resolve the same whether the api runs from apps/api or root.
_REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"

    database_url: str = "postgresql+psycopg://itai:change-me@localhost:5432/itai"
    redis_url: str = "redis://localhost:6379/0"

    api_secret_key: str = "change-me"
    api_cors_origins: str = "http://localhost:3000"

    # file-backed benchmark registry. swapped for a db-backed store in a later phase.
    benchmark_store_path: str = "var/benchmarks.json"

    # artifacts produced by the scripts (relative to the repo root). the read endpoints and
    # the prediction endpoint serve these; a later phase moves them into object storage + db.
    reports_dir: str = "reports"
    model_dir: str = "models/detector"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.api_cors_origins.split(",") if origin.strip()]

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else _REPO_ROOT / path

    @property
    def reports_path(self) -> Path:
        return self._resolve(self.reports_dir)

    @property
    def model_path(self) -> Path:
        return self._resolve(self.model_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
