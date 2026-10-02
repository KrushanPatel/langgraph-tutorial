"""Migration guide: deprecated LangGraph APIs next to their replacements.

You will meet these in older code and tutorials. Each one still works but
emits a subclass of `LangGraphDeprecationWarning`, so CI can catch them:

    filterwarnings = ["error::langgraph.warnings.LangGraphDeprecationWarning"]

| Deprecated (warning class)                         | Use instead                                          |
|----------------------------------------------------|------------------------------------------------------|
| `MessageGraph` (`SinceV10`)                        | `StateGraph(MessagesState)`                           |
| `prebuilt.create_react_agent` (`SinceV10`)         | `langchain.agents.create_agent`, or `ToolNode` + `tools_condition` (see `graphs/runbook_agent.py`) |
| `prebuilt.ValidationNode` (`SinceV10`)             | structured output / tool error handling               |
| `errors.NodeInterrupt` (`SinceV10`)                | `types.interrupt()`                                   |
| `StateGraph(input=..., output=...)` (`SinceV05`)   | `input_schema=` / `output_schema=`                    |
| `@task(retry=...)` (`SinceV05`)                    | `retry_policy=`                                       |
| `result["__interrupt__"]` on v2 output (`SinceV11`) | `GraphOutput.value` / `GraphOutput.interrupts`        |
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.errors import NodeInterrupt
from langgraph.graph import END, START, MessageGraph, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ValidationNode, create_react_agent
from langgraph.warnings import LangGraphDeprecationWarning
from pydantic import BaseModel

from lg_prod.context import DeskContext
from lg_prod.graphs.runbook_agent import AgentState


class IncidentTicket(BaseModel):
    """Schema the legacy extractor validates tool calls against."""

    service: str
    severity: int


def legacy_echo_graph() -> Any:
    """`MessageGraph`: the whole state is a list of messages."""
    builder = MessageGraph()

    def respond(messages: Sequence[AnyMessage]) -> AIMessage:
        return AIMessage(f"ack: {messages[-1].content}")

    builder.add_node("respond", respond)  # type: ignore[arg-type]
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile()


def modern_echo_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    def respond(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(f"ack: {state['messages'][-1].content}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    return builder.compile()


def legacy_react_agent(model: BaseChatModel, tools: Sequence[Any]) -> Any:
    # Tools using `InjectedState("service")` need a state schema with that
    # key; the default (messages + remaining_steps) would drop it.
    return create_react_agent(
        model, tools, state_schema=AgentState, context_schema=DeskContext, name="legacy_agent"
    )


def legacy_ticket_validator() -> Any:
    """`ValidationNode`: checks tool-call args against schemas without running anything."""
    builder = StateGraph(MessagesState)
    builder.add_node("validate", ValidationNode([IncidentTicket]))
    builder.add_edge(START, "validate")
    return builder.compile()


def legacy_interrupting_node(state: dict[str, Any]) -> dict[str, Any]:
    """Old-style interrupt: raise `NodeInterrupt` instead of calling `interrupt()`."""
    raise NodeInterrupt({"question": "approve?"})


@contextmanager
def deprecations_as_errors() -> Iterator[None]:
    """Turn every LangGraph deprecation into an error inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", LangGraphDeprecationWarning)
        yield
