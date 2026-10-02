"""Run-scoped context: *who* is running the graph, passed on every call.

LangGraph separates two kinds of data:

* **State** (`state.py`) is what the graph is *about*. It flows between nodes
  through channels and is persisted by the checkpointer, so it survives
  interrupts, crashes and time travel.
* **Context** (this module) is what the run is *for*: identity, tenant and
  permissions. It is declared with `StateGraph(..., context_schema=DeskContext)`,
  supplied with `graph.invoke(..., context=DeskContext(...))`, read inside nodes
  as `runtime.context`, and is **never checkpointed**.

Keep authorization data in context. If it lived in state, a client could
forge it through `update_state`, and an old checkpoint would replay a user's
stale permissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["engineer", "oncall_lead"]


@dataclass(frozen=True, slots=True)
class DeskContext:
    user_id: str
    role: Role = "engineer"
    team: str = "platform"

    @property
    def can_approve(self) -> bool:
        """Only on-call leads may approve risky remediations."""
        return self.role == "oncall_lead"

    @property
    def memory_namespace(self) -> tuple[str, ...]:
        """Store namespace for long-term memory, scoped per team."""
        return ("teams", self.team, "incidents")
