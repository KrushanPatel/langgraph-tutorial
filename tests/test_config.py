"""Settings: .env handling, validation, and model construction."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lg_prod.config import Settings
from lg_prod.models import build_chat_model


def test_empty_values_in_env_file_mean_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LANGGRAPH_AES_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text((Path(__file__).parent.parent / ".env.example").read_text())

    settings = Settings(_env_file=env)  # type: ignore[call-arg]

    assert settings.checkpoint_aes_key is None
    assert settings.persistence == "memory"


def test_bad_aes_key_fails_at_startup():
    with pytest.raises(ValidationError, match="16, 24 or 32 bytes"):
        Settings(_env_file=None, LANGGRAPH_AES_KEY="short")  # type: ignore[call-arg]


def test_api_key_from_settings_reaches_the_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(_env_file=None, ANTHROPIC_API_KEY="sk-test")  # type: ignore[call-arg]

    model = build_chat_model(settings)

    assert model.anthropic_api_key.get_secret_value() == "sk-test"  # type: ignore[attr-defined]
