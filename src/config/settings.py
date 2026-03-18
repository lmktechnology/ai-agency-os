from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class GlobalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === LLM Provider API Keys ===
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    deepseek_api_key: str = ""
    ollama_base_url: str = ""   # e.g. http://localhost:11434/v1

    # === Telegram ===
    telegram_bot_token: str = ""
    telegram_allowed_chat_ids: list[int] = []

    # === Directories ===
    data_dir: str = "./memory"
    agents_dir: str = "./agents"
    skills_dir: str = "./skills"
    tools_dir: str = "./tools"

    log_level: str = "INFO"
    vector_store_path: str = "./memory/vector"

    telegram_admin_chat_id: str = ""
    webhook_url: str = ""
    webhook_port: int = 8443
