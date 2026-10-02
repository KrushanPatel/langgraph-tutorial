"""Graph state: the schema, its channels and their reducers.

Every key of a `StateGraph` state schema becomes a **channel**, and the
annotation decides which one:

| Annotation                                   | Channel                     | Semantics                         |
|----------------------------------------------|-----------------------------|-----------------------------------|
| `str` (no reducer)                           | `LastValue`                 | one writer per step, else error   |
| `Annotated[list, operator.add]`              | `BinaryOperatorAggregate`   | fold every write with the reducer |
| `Annotated[list, add_messages]`              | `BinaryOperatorAggregate`   | merge by message id, remove, ...  |
| `Annotated[str, EphemeralValue(str)]`        | `EphemeralValue`            | visible for one step only         |
| `Annotated[str, UntrackedValue(str)]`        | `UntrackedValue`            | never checkpointed                |
| `Annotated[list, DeltaChannel(fn)]`          | `DeltaChannel` (beta)       | checkpoints deltas, not snapshots |
| `RemainingSteps` / `IsLastStep`              | managed value               | computed by the runtime, read-only|

The graph has three schemas, which is the production pattern for a clean API:

* `IncidentInput` - what callers may send (`input_schema=`).
* `IncidentOutput` - what callers get back (`output_schema=`).
* `IncidentState` - everything nodes share, a superset of both.

`Scratchpad` is a *private* schema. Its keys are deliberately absent from
`IncidentState`: the graph registers them only because a node declares
`input_schema=Scratchpad`. Nodes can pass data through them, but they are
not part of the input or output, so they never leak to the caller.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.channels import DeltaChannel, EphemeralValue, UntrackedValue
from langgraph.graph.message import add_messages
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer
from langgraph.managed import IsLastStep, RemainingSteps

Severity = Literal["sev1", "sev2", "sev3", "sev4"]
Category = Literal["latency", "errors", "capacity", "security", "unknown"]
Status = Literal["open", "remediated", "escalated", "closed", "rejected"]

SEVERITY_RANK: dict[Severity, int] = {"sev1": 1, "sev2": 2, "sev3": 3, "sev4": 4}


class Finding(TypedDict):
    check: str
    ok: bool
    detail: str


def merge_findings(left: list[Finding] | None, right: list[Finding] | None) -> list[Finding]:
    """Custom reducer: keyed upsert by `check`, so a re-run check replaces its old row.

    A plain `operator.add` would append duplicates every time a diagnostic is
    retried or re-run after time travel. Reducers must be pure and must accept
    `None` for the left side (the channel's empty value).
    """
    merged: dict[str, Finding] = {f["check"]: f for f in left or []}
    for finding in right or []:
        merged[finding["check"]] = finding
    return list(merged.values())


def append_audit(state: list[str] | None, writes: Sequence[list[str]]) -> list[str]:
    """`DeltaChannel` reducer: `(state, [write, ...]) -> state`.

    It must be batching-invariant, because LangGraph replays checkpointed
    writes in bigger batches than they were written:
    `f(f(s, xs), ys) == f(s, xs + ys)`. Concatenation satisfies this.
    """
    out = list(state or [])
    for batch in writes:
        out.extend(batch)
    return out


class IncidentInput(TypedDict):
    report: str
    service: str


class IncidentOutput(TypedDict, total=False):
    incident_id: str
    severity: Severity
    status: Status
    summary: str
    actions_taken: list[str]


class Scratchpad(TypedDict, total=False):
    """Private node-to-node data. Not in the input or output schema."""

    runbook: str
    similar_incidents: list[str]


class IncidentState(TypedDict, total=False):
    # --- From the caller --------------------------------------------------
    report: str
    service: str

    # --- Conversation with the LLM-backed nodes ---------------------------
    messages: Annotated[list[AnyMessage], add_messages]

    # --- Plain LastValue channels: exactly one writer per superstep -------
    incident_id: str
    severity: Severity
    category: Category
    status: Status
    summary: str
    approved_by: str | None

    # --- Reducer channels: any number of writers per superstep ------------
    findings: Annotated[list[Finding], merge_findings]
    actions_taken: Annotated[list[str], operator.add]
    timeline: Annotated[list[str], operator.add]

    # --- Special channels ---------------------------------------------------
    # Shown to the next step only, then cleared. Good for one-shot signals.
    handoff_note: Annotated[str, EphemeralValue(str)]
    # Lives in memory for the current run; never written to a checkpoint.
    # Use for per-run handles that must not be persisted (and are lost on resume).
    trace_url: Annotated[str, UntrackedValue(str)]
    # Long append-only log stored as deltas instead of full copies per step.
    audit: Annotated[list[str], DeltaChannel(append_audit)]

    # Generative-UI messages (`push_ui_message`), merged by id.
    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]

    # --- Managed values: computed by the runtime, never written -----------
    remaining_steps: RemainingSteps
    is_last_step: IsLastStep


class CheckInput(TypedDict):
    """Payload of one `Send` in the diagnostics fan-out."""

    service: str
    check: str
