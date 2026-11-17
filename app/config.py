import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI-Based Viva Interview System"
    app_version: str = "1.0.0"
    max_questions: int = 3
    port: int = 8080
    
    # LLM Settings
    llm_provider: str = "openai"  # "openai", "gemini", or "mock"
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"
    openai_model: str = "gpt-4o-mini"
    
    # Environment file config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
