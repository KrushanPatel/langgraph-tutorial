"""Calling a graph deployed on LangGraph Server (Agent Server) with `RemoteGraph`.

`RemoteGraph` implements the same protocol as a local compiled graph
(`invoke`, `stream`, `get_state`, ...), but every call goes over HTTP via the
`langgraph_sdk` client. That gives you two production patterns:

* **Client.** A web backend calls a deployed graph exactly as it would call a
  local one.
* **Composition.** A `RemoteGraph` can be added as a *node* of a local graph,
  so a small local "front door" graph can delegate to a remote agent owned by
  another team.

Errors raised by the remote run arrive as an `error` stream event, which
`RemoteGraph` re-raises locally as `RemoteException`.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel.remote import RemoteGraph


def connect_runbook_agent(
    *,
    url: str | None = None,
    api_key: str | None = None,
    sync_client: Any | None = None,
) -> RemoteGraph:
    """`url` for a real deployment; `sync_client` to inject a client (tests)."""
    return RemoteGraph(
        "runbook_agent",  # assistant id or graph name in langgraph.json
        url=url,
        api_key=api_key,
        sync_client=sync_client,
        name="remote_runbook_agent",
        distributed_tracing=True,  # propagate LangSmith trace headers
    )


class FrontDoorState(TypedDict, total=False):
    messages: Annotated[list[Any], operator.add]
    service: str
    notes: Annotated[list[str], operator.add]
    routed_to: str


def build_front_door(remote: RemoteGraph) -> CompiledStateGraph[Any, Any, Any, Any]:
    """A local graph that labels the request and delegates to the remote agent."""

    def label(state: FrontDoorState) -> FrontDoorState:
        return {"routed_to": remote.name or remote.assistant_id}

    builder = StateGraph(FrontDoorState)
    builder.add_node("label", label)
    builder.add_node("remote", remote)
    builder.add_edge(START, "label")
    builder.add_edge("label", "remote")
    return builder.compile()
