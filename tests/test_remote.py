"""RemoteGraph against a loopback client that runs a local graph in-process."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, message_to_dict, messages_from_dict
from langgraph.pregel.remote import RemoteException
from langgraph_sdk.schema import StreamPart

from lg_prod.context import DeskContext
from lg_prod.graphs.runbook_agent import build_runbook_agent
from lg_prod.remote import build_front_door, connect_runbook_agent
from tests.conftest import call, script


def _wire(value: Any) -> Any:
    """Serialize like the HTTP API would (messages become dicts)."""
    if isinstance(value, dict):
        return {k: _wire(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_wire(v) for v in value]
    if hasattr(value, "model_dump") and hasattr(value, "type"):
        return message_to_dict(value)["data"] | {"type": value.type}
    return value


@dataclass
class _Runs:
    graph: Any
    fail_with: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def stream(self, **kwargs: Any) -> Iterator[StreamPart]:
        self.calls.append(kwargs)
        if self.fail_with:
            yield StreamPart("error", {"error": "ValueError", "message": self.fail_with})
            return
        payload = dict(kwargs["input"])
        # The real SDK JSON-encodes message objects on the wire; accept both.
        payload["messages"] = [
            m
            if not isinstance(m, dict)
            else messages_from_dict([{"type": m["type"], "data": m}])[0]
            for m in payload.get("messages", [])
        ] or [HumanMessage("hi")]
        ctx = DeskContext(**(kwargs["context"] or {"user_id": "remote"}))
        for mode, data in self.graph.stream(
            payload, context=ctx, stream_mode=kwargs["stream_mode"]
        ):
            yield StreamPart(mode, _wire(data))


@dataclass
class LoopbackClient:
    """Duck-typed `SyncLangGraphClient`: only what RemoteGraph.stream needs."""

    runs: _Runs


def _client(services, fail_with: str | None = None) -> LoopbackClient:  # type: ignore[no-untyped-def]
    agent = build_runbook_agent(
        script(call("run_diagnostic", {"check": "db"}), "db is down"), services
    )
    return LoopbackClient(_Runs(agent, fail_with))


def test_remote_graph_invoke_over_the_sdk_protocol(services):  # type: ignore[no-untyped-def]
    client = _client(services)
    remote = connect_runbook_agent(sync_client=client)

    out = remote.invoke(
        {"messages": [HumanMessage("why?")], "service": "checkout", "notes": []},
        context={"user_id": "u-lead", "role": "oncall_lead"},
    )

    assert out["messages"][-1]["content"] == "db is down"
    [request] = client.runs.calls
    assert request["assistant_id"] == "runbook_agent"
    assert request["context"] == {"user_id": "u-lead", "role": "oncall_lead"}


def test_remote_graph_as_a_node_in_a_local_graph(services):  # type: ignore[no-untyped-def]
    front = build_front_door(connect_runbook_agent(sync_client=_client(services)))

    out = front.invoke({"messages": [], "service": "checkout", "notes": []})

    assert out["routed_to"] == "remote_runbook_agent"
    assert out["messages"][-1]["content"] == "db is down"


def test_remote_errors_surface_as_remote_exception(services):  # type: ignore[no-untyped-def]
    remote = connect_runbook_agent(sync_client=_client(services, fail_with="boom"))

    with pytest.raises(RemoteException, match="boom"):
        remote.invoke({"messages": [], "service": "x", "notes": []})


def test_remote_graph_requires_a_client():
    remote = connect_runbook_agent()
    with pytest.raises(ValueError, match="Sync client is not initialized"):
        remote.invoke({})
