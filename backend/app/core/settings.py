from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Helios-Grid Backend"
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    debug: bool = True
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    cors_allow_methods: str = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
    cors_allow_headers: str = "*"
    cors_allow_credentials: bool = True
    # Do NOT hardcode production DB URLs or credentials here. Prefer env / secret backends.
    database_url: str | None = None
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10
    # Redis URL may be provided via env or a secret backend. None => no redis configured.
    redis_url: str | None = None
    redis_decode_responses: bool = True
    ray_address: str = ""
    ray_namespace: str = "helios-grid"
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "helios-grid"
    mlflow_log_models: bool = True
    model_registry_enabled: bool = False
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "us-east-1"
    s3_use_ssl: bool = True
    s3_force_path_style: bool = True
    s3_prefix: str = "models"
    simulation_ws_enabled: bool = True
    simulation_pubsub_channel: str = "simulations.updates"
    auth_enabled: bool = True
    # JWT secret should not be hardcoded. Prefer `JWT_SECRET_KEY` in environment or a secret backend.
    jwt_secret_key: str | None = None
    jwt_algorithm: str = "HS256"
    access_token_expires_minutes: int = 60
    refresh_token_expires_days: int = 7
    refresh_token_rotate: bool = True
    # Rate limiting
    rate_limit_default: str = "1000/hour"
    rate_limit_auth: str = "10/minute"
    rate_limit_simulation: str = "60/minute"

    @property
    def effective_rate_limit_default(self) -> str:
        # In test mode, lift the rate-limit ceiling so test suites that
        # register many users (or fire many requests per endpoint) are
        # not throttled by slowapi.
        return "1000000/second" if self.app_env == "test" else self.rate_limit_default

    @property
    def effective_rate_limit_auth(self) -> str:
        return "1000000/second" if self.app_env == "test" else self.rate_limit_auth

    @property
    def effective_rate_limit_simulation(self) -> str:
        return "1000000/second" if self.app_env == "test" else self.rate_limit_simulation
    # Secret backend configuration (optional)
    secret_backend: str = "env"  # env | vault | doppler
    vault_addr: str | None = None
    vault_token: str | None = None
    doppler_token: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def parsed_cors_allow_origins(self) -> list[str]:
        origins = [
            origin.strip()
            for origin in self.cors_allow_origins.split(",")
            if origin.strip()
        ]
        return origins or ["http://localhost:3000"]

    @property
    def parsed_cors_allow_methods(self) -> list[str]:
        methods = [
            method.strip().upper()
            for method in self.cors_allow_methods.split(",")
            if method.strip()
        ]
        return methods or ["GET", "POST"]

    @property
    def parsed_cors_allow_headers(self) -> list[str]:
        headers = [
            header.strip()
            for header in self.cors_allow_headers.split(",")
            if header.strip()
        ]
        return headers or ["*"]


settings = Settings()

__all__ = ["Settings", "settings"]
