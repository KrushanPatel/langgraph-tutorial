"""Typed, environment-driven configuration.

Production rule: no model names, limits, paths or secrets hard-coded in graph
code. Everything tunable lives here, is validated once at startup, and is
injected into the graph factories (`graphs.*`, `persistence`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        # `KEY=` in .env means "unset", not "empty string" (an empty AES key
        # would otherwise crash on the first checkpoint write).
        env_ignore_empty=True,
    )

    # --- Models (only the LLM-backed nodes use these) ----------------------
    chat_model: str = Field(
        default="anthropic:claude-sonnet-5",
        description="`provider:model` string understood by `init_chat_model`.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    request_timeout_s: float = Field(default=60.0, gt=0)

    # --- Graph execution ----------------------------------------------------
    recursion_limit: int = Field(
        default=50,
        gt=0,
        description="Max supersteps per run. Each node execution is one step.",
    )
    durability: Literal["sync", "async", "exit"] = Field(
        default="async",
        description="When checkpoints are written: before the next step, in parallel, or at exit.",
    )
    node_max_attempts: int = Field(default=3, ge=1)
    metrics_timeout_s: float = Field(default=5.0, gt=0)
    classification_cache_ttl_s: int = Field(default=600, gt=0)

    # --- Persistence --------------------------------------------------------
    persistence: Literal["memory", "sqlite"] = "memory"
    sqlite_path: str = "incident_desk.sqlite"
    # 16/24/32-byte AES key. When set, checkpoints are encrypted at rest.
    checkpoint_aes_key: SecretStr | None = Field(default=None, alias="LANGGRAPH_AES_KEY")
    memory_ttl_minutes: float = Field(default=60 * 24 * 30, gt=0)

    @field_validator("checkpoint_aes_key")
    @classmethod
    def _aes_key_length(cls, key: SecretStr | None) -> SecretStr | None:
        if key is not None and len(key.get_secret_value().encode()) not in (16, 24, 32):
            raise ValueError("LANGGRAPH_AES_KEY must be 16, 24 or 32 bytes")
        return key

    # --- Policy -------------------------------------------------------------
    auto_remediate_max_severity: Literal["sev1", "sev2", "sev3", "sev4"] = Field(
        default="sev3",
        description="Remediations at this severity or milder run without human approval.",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Tests build `Settings(...)` directly."""
    return Settings()
