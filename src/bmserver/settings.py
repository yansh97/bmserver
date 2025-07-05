from typing import ClassVar

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="BMSERVER_"
    )

    postgres_url: str | None = None


load_dotenv()
settings = Settings()
