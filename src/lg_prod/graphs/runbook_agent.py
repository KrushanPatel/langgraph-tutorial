"""A tool-calling agent built by hand from LangGraph's prebuilt pieces.

`langchain.agents.create_agent` is the batteries-included way to build an
agent. This module builds the same loop by hand, so each part is visible:

    START -> agent --tools_condition--> tools -> agent ... -> END

* `MessagesState` - a ready-made state with an `add_messages` reducer.
* `ToolNode` - runs every tool call in the last `AIMessage` in parallel and
  returns one `ToolMessage` per call (or `Command`s from tools).
* `tools_condition` - routes to `"tools"` when the last message has tool
  calls, else to `END`.
* Injection: arguments the model must never see or forge.
    - `InjectedState("service")` - one state key (or the whole state).
    - `InjectedStore()` - the graph's `BaseStore`.
    - `InjectedToolCallId` - needed to answer with a `Command`.
    - `ToolRuntime` - all of the above plus `context`, `config`,
      `stream_writer`, `execution_info` and `emit_output_delta`.
* `handle_tool_errors=` - which exceptions become an error `ToolMessage`
  the model can react to; anything else fails the run.
* `wrap_tool_call=` - an interceptor around every tool execution (audit,
  policy, retries), the ToolNode-level equivalent of middleware.
* `RemainingSteps` - stop the loop gracefully before `GraphRecursionError`.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, ToolException, tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import InjectedState, InjectedStore, ToolNode, ToolRuntime, tools_condition
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Command

from lg_prod.context import DeskContext
from lg_prod.services import Services

SYSTEM_PROMPT = (
    "You are an SRE assistant. Diagnose the service with the tools, take notes, "
    "and only restart services when diagnostics justify it."
)


class AgentState(MessagesState):
    service: str
    notes: Annotated[list[str], operator.add]
    remaining_steps: RemainingSteps


def build_tools(services: Services) -> list[Any]:
    @tool
    def run_diagnostic(
        check: Literal["db", "cache", "network", "cpu", "memory", "disk"],
        service: Annotated[str, InjectedState("service")],
    ) -> str:
        """Run one health check against the incident's service."""
        finding = services.diagnostics.run(service, check)
        return f"{check}: {'OK' if finding['ok'] else 'FAILING'} ({finding['detail']})"

    @tool
    def recall_incidents(
        query: str,
        store: Annotated[BaseStore, InjectedStore()],
        state: Annotated[dict[str, Any], InjectedState],
    ) -> str:
        """Search past incidents for this service."""
        hits = store.search(("teams", "platform", "incidents"), query=query, limit=3)
        relevant = [h.value["summary"] for h in hits if h.value["service"] == state["service"]]
        return "\n".join(relevant) or "no similar incidents"

    @tool
    def add_note(note: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:  # type: ignore[type-arg]
        """Record a finding in the incident notes."""
        # A tool returning `Command` updates graph state directly. It must also
        # include the ToolMessage answering its own tool call.
        return Command(
            update={
                "notes": [note],
                "messages": [ToolMessage("noted", tool_call_id=tool_call_id)],
            }
        )

    @tool
    def restart_service(reason: str, runtime: ToolRuntime[DeskContext, AgentState]) -> str:
        """Restart the incident's service. On-call leads only."""
        if not runtime.context.can_approve:
            # PermissionError is in `handle_tool_errors`: the model sees it.
            raise PermissionError(f"{runtime.context.user_id} may not restart services")
        service = runtime.state["service"]
        for pct in (0, 50, 100):
            runtime.emit_output_delta({"progress": pct})  # `tools` stream channel
        runtime.stream_writer({"event": "restart", "service": service})  # `custom`
        return services.remediator.apply(service, f"restart ({reason})")

    @tool
    def page_human(message: str) -> str:
        """Page the on-call human."""
        if not message.strip():
            raise ToolException("message must not be empty")
        return services.pager.page("platform", message)

    return [run_diagnostic, recall_incidents, add_note, restart_service, page_human]


def audit_tool_calls(log: list[str]) -> Callable[..., ToolMessage | Command]:  # type: ignore[type-arg]
    """`wrap_tool_call` interceptor: records every call and its outcome."""

    def wrapper(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],  # type: ignore[type-arg]
    ) -> ToolMessage | Command:  # type: ignore[type-arg]
        result = execute(request)
        status = getattr(result, "status", "command")
        log.append(f"{request.tool_call['name']}:{status}")
        return result

    return wrapper


def build_runbook_agent(
    model: BaseChatModel,
    services: Services,
    *,
    checkpointer: Checkpointer = None,
    store: BaseStore | None = None,
    audit_log: list[str] | None = None,
) -> CompiledStateGraph[AgentState, DeskContext, AgentState, AgentState]:
    tools = build_tools(services)
    bound = model.bind_tools(tools)

    def agent(state: AgentState, runtime: Runtime[DeskContext]) -> dict[str, Any]:
        if state["remaining_steps"] < 3:
            return {"messages": [AIMessage("Step budget exhausted; handing over to a human.")]}
        prompt = SystemMessage(f"{SYSTEM_PROMPT} Operator: {runtime.context.user_id}.")
        return {"messages": [bound.invoke([prompt, *state["messages"]])]}

    tool_node = ToolNode(
        tools,
        handle_tool_errors=(PermissionError, ToolException),
        wrap_tool_call=audit_tool_calls(audit_log if audit_log is not None else []),
    )

    builder = StateGraph(AgentState, context_schema=DeskContext)
    builder.add_node("agent", agent)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer, store=store, name="runbook_agent")
