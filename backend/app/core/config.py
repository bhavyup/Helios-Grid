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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def parsed_cors_allow_origins(self) -> list[str]:
        origins = [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]
        return origins or ["http://localhost:3000"]

    @property
    def parsed_cors_allow_methods(self) -> list[str]:
        methods = [method.strip().upper() for method in self.cors_allow_methods.split(",") if method.strip()]
        return methods or ["GET", "POST"]

    @property
    def parsed_cors_allow_headers(self) -> list[str]:
        headers = [header.strip() for header in self.cors_allow_headers.split(",") if header.strip()]
        return headers or ["*"]


settings = Settings()
