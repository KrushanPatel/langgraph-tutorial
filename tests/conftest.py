"""Test harness: real graphs, real checkpointers and stores; only LLMs are faked.

`ScriptedChatModel` replays scripted `AIMessage`s, records what it was sent,
and streams tool calls (the stock `GenericFakeChatModel._stream` drops them,
which breaks `stream_mode="messages"` and `stream_events(version="v3")`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field

from lg_prod.config import Settings
from lg_prod.context import DeskContext
from lg_prod.graphs.triage import IncidentGraph, build_incident_graph
from lg_prod.services import Services, demo_services


class ScriptedChatModel(GenericFakeChatModel):
    """Replays AIMessages in order and records what the graph sent it."""

    bound_tools: list[list[str]] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:
        self.bound_tools.append([getattr(t, "name", None) or t.get("name") for t in tools])
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    def _stream(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        message = self._generate(messages, *args, **kwargs).generations[0].message
        for word in re.split(r"(\s)", message.text) if message.text else []:
            yield ChatGenerationChunk(message=AIMessageChunk(content=word, id=message.id))
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                id=message.id,
                tool_call_chunks=[
                    {
                        "name": tc["name"],
                        "args": json.dumps(tc["args"]),
                        "id": tc["id"],
                        "index": i,
                        "type": "tool_call_chunk",
                    }
                    for i, tc in enumerate(message.tool_calls)
                ],
                usage_metadata=message.usage_metadata,
                chunk_position="last",
            )
        )


def script(*turns: AIMessage | str) -> ScriptedChatModel:
    it: Iterator[AIMessage | str] = iter(turns)
    return ScriptedChatModel(messages=it)


def call(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def thread(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, metrics_timeout_s=0.5)  # type: ignore[call-arg]


@pytest.fixture
def services() -> Services:
    return demo_services()


@pytest.fixture
def lead() -> DeskContext:
    return DeskContext(user_id="u-lead", role="oncall_lead")


@pytest.fixture
def engineer() -> DeskContext:
    return DeskContext(user_id="u-eng", role="engineer")


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def desk(settings: Settings, services: Services, store: InMemoryStore) -> IncidentGraph:
    return build_incident_graph(
        settings, services, checkpointer=InMemorySaver(), store=store, cache=InMemoryCache()
    )
