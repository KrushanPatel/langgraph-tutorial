"""Escalation subgraph: pages the owning team, then hands control back to the parent.

Subgraph patterns shown here:

* **Different state schema.** `EscalationState` shares only some keys with
  the parent's `IncidentState`. When a compiled graph is added as a node,
  LangGraph passes the shared keys in and maps the shared keys back out.
* **`Command(graph=Command.PARENT, ...)`** lets a subgraph node jump to a
  node in the *parent* graph. Inside, LangGraph raises `ParentCommand`
  (a `GraphBubbleUp`), which the parent catches and applies. Never catch
  `GraphBubbleUp` subclasses in your own `except Exception` blocks: let them
  propagate, or interrupts and parent commands stop working.
* **Checkpointer inheritance.** Compiling with `checkpointer=None` (the
  `Checkpointer` default) makes the subgraph use the parent's checkpointer,
  so an `interrupt()` inside it pauses and resumes the whole run.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer, Command, interrupt

from lg_prod.context import DeskContext
from lg_prod.services import Services
from lg_prod.state import Severity


class EscalationState(TypedDict, total=False):
    service: str
    severity: Severity
    summary: str
    page_id: str
    acknowledged_by: str
    actions_taken: Annotated[list[str], operator.add]
    timeline: Annotated[list[str], operator.add]


class AckRequest(TypedDict):
    """Shape of the resume value expected by `wait_for_ack`."""

    acknowledged_by: str


def build_escalation_graph(
    services: Services, *, checkpointer: Checkpointer = None
) -> CompiledStateGraph[EscalationState, DeskContext, EscalationState, EscalationState]:
    def page_team(state: EscalationState, runtime: Runtime[DeskContext]) -> EscalationState:
        page_id = services.pager.page(
            runtime.context.team,
            f"[{state.get('severity', 'sev?')}] {state['service']}: {state.get('summary', '')}",
        )
        return {"page_id": page_id, "timeline": [f"paged {runtime.context.team} ({page_id})"]}

    def wait_for_ack(state: EscalationState) -> EscalationState:
        # Sev1 pages need a human to acknowledge before the incident closes.
        if state.get("severity") != "sev1":
            return {"acknowledged_by": "auto"}
        answer = interrupt(
            {"type": "page_ack", "page_id": state["page_id"]},
            response_schema=AckRequest,
        )
        return {"acknowledged_by": answer["acknowledged_by"]}

    def hand_back(state: EscalationState) -> Command:  # type: ignore[type-arg]
        # Jump straight to the parent's `close` node. The update is applied to
        # the *parent* state, so it may only use keys the parent has.
        # Do NOT annotate this as `Command[Literal["close"]]`: return-type
        # literals declare edges of *this* graph, and `close` isn't in it.
        return Command(
            graph=Command.PARENT,
            goto="close",
            update={
                "status": "escalated",
                "actions_taken": [f"escalated via {state['page_id']}"],
                "timeline": [f"acknowledged by {state.get('acknowledged_by')}"],
            },
        )

    builder = StateGraph(EscalationState, context_schema=DeskContext)
    builder.add_node(page_team)
    builder.add_node(wait_for_ack)
    builder.add_node(hand_back)
    builder.add_edge(START, "page_team")
    builder.add_edge("page_team", "wait_for_ack")
    builder.add_edge("wait_for_ack", "hand_back")
    builder.add_edge("hand_back", END)
    return builder.compile(checkpointer=checkpointer, name="escalation")
