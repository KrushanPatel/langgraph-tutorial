"""LLM access for the few nodes that need one.

Most of a production LangGraph app is deterministic code; the LLM sits behind
a narrow interface (`Classifier`) so it can be swapped for rules, a cheaper
model or a test fake without touching the graph.

`TAG_NOSTREAM`: the classifier's LLM call is an internal step. Tagging the
model with it keeps its tokens out of `stream_mode="messages"`, so the user
interface only streams the replies meant for humans.
"""

from __future__ import annotations

from typing import Any, Protocol

from langchain.chat_models import BaseChatModel, init_chat_model
from langgraph.constants import TAG_NOSTREAM
from pydantic import BaseModel, Field

from lg_prod.config import Settings
from lg_prod.state import Category, Severity


class Classification(BaseModel):
    severity: Severity
    category: Category
    rationale: str = Field(max_length=500)


class Classifier(Protocol):
    def classify(self, report: str, service: str) -> Classification: ...


class RuleClassifier:
    """Deterministic keyword rules. The default, and the fallback when the LLM fails."""

    _CATEGORIES: tuple[tuple[Category, tuple[str, ...]], ...] = (
        ("security", ("breach", "leak", "unauthorized", "cve")),
        ("errors", ("500", "error", "exception", "failing", "down")),
        ("latency", ("slow", "latency", "timeout", "p99")),
        ("capacity", ("cpu", "memory", "disk", "oom", "saturat")),
    )

    def classify(self, report: str, service: str) -> Classification:
        text = report.lower()
        category: Category = next(
            (cat for cat, words in self._CATEGORIES if any(w in text for w in words)),
            "unknown",
        )
        if category == "security" or "outage" in text or "down" in text:
            severity: Severity = "sev1"
        elif "customers" in text or category == "errors":
            severity = "sev2"
        elif category in ("latency", "capacity"):
            severity = "sev3"
        else:
            severity = "sev4"
        return Classification(severity=severity, category=category, rationale="keyword rules")


class LLMClassifier:
    """Structured-output classification with any LangChain chat model."""

    def __init__(self, model: BaseChatModel) -> None:
        self._chain = model.with_structured_output(Classification).with_config(
            tags=[TAG_NOSTREAM], run_name="classify_incident"
        )

    def classify(self, report: str, service: str) -> Classification:
        result: Any = self._chain.invoke(
            [
                ("system", "Classify the incident. Be conservative: prefer higher severity."),
                ("user", f"service={service}\n\n{report}"),
            ]
        )
        return result if isinstance(result, Classification) else Classification(**result)


def build_chat_model(settings: Settings) -> BaseChatModel:
    # pydantic-settings reads `.env` into Settings only, not os.environ, so a
    # key that lives in `.env` must be passed on explicitly.
    extra: dict[str, Any] = {}
    if settings.anthropic_api_key is not None and settings.chat_model.startswith("anthropic:"):
        extra["api_key"] = settings.anthropic_api_key
    model: BaseChatModel = init_chat_model(
        settings.chat_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.request_timeout_s,
        max_retries=2,
        **extra,
    )
    return model
