"""Alert correlation pipeline written directly on Pregel: channels + `NodeBuilder`.

`StateGraph` and `@entrypoint` both compile to a `Pregel` object. Working on
Pregel directly shows the actual execution model:

* Execution proceeds in **supersteps**. In each one, every node whose
  *subscribed* channels changed in the previous step runs in parallel. Their
  writes are applied together at the end of the step, through each channel's
  update rule.
* A node **subscribes** to channels (which trigger it), **reads** others
  without being triggered by them, and **writes** to channels.

Every built-in channel is used once here, plus a custom one:

| Channel                         | Used for                                         |
|---------------------------------|--------------------------------------------------|
| `Topic`                         | `alerts` (per-step batch), `seen` (accumulating) |
| `BinaryOperatorAggregate`       | `count`: running total with `operator.add`       |
| `MaxValue` (custom, below)      | `peak`: max severity score so far                |
| `AnyValue`                      | `status`: many writers allowed; cleared in any step with no writes |
| `NamedBarrierValue`             | `joined`: fires `correlate` once BOTH `dedupe` and `score` wrote |
| `LastValue`                     | `digest`: the output                             |
| `EphemeralValue`                | `tick`: a one-step trigger                       |
| `LastValueAfterFinish`          | `report`: seen only after the run would end      |
| `NamedBarrierValueAfterFinish`  | `archive_gate`: barrier that also waits for the end |
| `UntrackedValue`                | `debug_scratch`: never checkpointed              |
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any, Self, TypedDict

from langgraph.channels import (
    AnyValue,
    BaseChannel,
    BinaryOperatorAggregate,
    EphemeralValue,
    LastValue,
    LastValueAfterFinish,
    NamedBarrierValue,
    NamedBarrierValueAfterFinish,
    Topic,
    UntrackedValue,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import TAG_HIDDEN
from langgraph.errors import EmptyChannelError, ErrorCode, InvalidUpdateError, create_error_message
from langgraph.pregel import NodeBuilder, Pregel
from langgraph.types import RetryPolicy

SEVERITY_SCORE = {"critical": 1.0, "warning": 0.5, "info": 0.1}


class Alert(TypedDict):
    id: str
    source: str
    level: str


class MaxValue(BaseChannel[float, float, float]):
    """Custom channel: keeps the maximum of every value ever written.

    The channel contract:
    * `get()` raises `EmptyChannelError` until the first write.
    * `update(values)` receives all writes of one superstep, returns
      whether the value changed (which is what triggers subscribers).
    * `checkpoint()` / `from_checkpoint()` make it durable.
    """

    __slots__ = ("value",)

    def __init__(self, typ: type[float] = float, key: str = "") -> None:
        super().__init__(typ, key)
        self.value: float | None = None

    @property
    def ValueType(self) -> type[float]:
        return float

    @property
    def UpdateType(self) -> type[float]:
        return float

    def copy(self) -> Self:
        new = self.__class__(self.typ, self.key)
        new.value = self.value
        return new

    def checkpoint(self) -> float | None:
        return self.value

    def from_checkpoint(self, checkpoint: Any) -> Self:
        new = self.__class__(self.typ, self.key)
        # On a fresh thread LangGraph passes a private `MISSING` sentinel, not
        # None, so accept only real values.
        new.value = float(checkpoint) if isinstance(checkpoint, int | float) else None
        return new

    def get(self) -> float:
        if self.value is None:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not None

    def update(self, values: Sequence[float]) -> bool:
        if not values:
            return False
        if any(v < 0 for v in values):
            raise InvalidUpdateError(
                create_error_message(
                    message=f"{self.key}: scores must be >= 0, got {list(values)}",
                    error_code=ErrorCode.INVALID_GRAPH_NODE_RETURN_VALUE,
                )
            )
        new = max(values) if self.value is None else max(self.value, *values)
        changed = new != self.value
        self.value = new
        return changed


# ------------------------------------------------------------------- nodes


def dedupe(inputs: dict[str, Any]) -> dict[str, Any]:
    seen = set(inputs.get("seen") or [])
    fresh = [a for a in inputs["alerts"] if a["id"] not in seen]
    return {"ids": [a["id"] for a in fresh], "count": len(fresh)}


def score(alerts: list[Alert]) -> float:
    return max((SEVERITY_SCORE.get(a["level"], 0.0) for a in alerts), default=0.0)


def correlate(inputs: dict[str, Any]) -> str:
    level = "page" if inputs["peak"] >= 1.0 else "ticket" if inputs["peak"] >= 0.5 else "log"
    return f"{inputs['count']} unique alert(s), peak={inputs['peak']:g} -> {level}"


def build_alert_pipeline(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Pregel[Any, Any, Any, Any]:
    nodes = {
        "dedupe": (
            NodeBuilder()
            .subscribe_to("alerts")  # triggers on new alerts...
            .read_from("seen")  # ...and reads history without being triggered by it
            .do(dedupe)
            .write_to(
                seen=lambda out: out["ids"] or None,  # mapper; None = skip write
                count=lambda out: out["count"],
                joined="dedupe",  # constant write
                status="correlating",
                archive_gate="dedupe",
            )
            .meta("pipeline", stage="ingest")
        ),
        "score": (
            NodeBuilder()
            .subscribe_only("alerts")  # input is the channel value itself
            .do(score)
            .write_to("peak", joined="score", status="correlating", archive_gate="score")
            .add_retry_policies(RetryPolicy(max_attempts=2))
        ),
        "correlate": (
            NodeBuilder()
            .subscribe_to("joined", read=False)  # trigger only; payload irrelevant
            .read_from("count", "peak")
            .do(correlate)
            .write_to("digest", "report", tick="go")
        ),
        # TAG_HIDDEN: plumbing node, left out of `updates`/`debug` stream output.
        "notify": NodeBuilder()
        .subscribe_only("tick")
        .do(lambda t: f"notified:{t}")
        .write_to("debug_scratch")
        .meta(TAG_HIDDEN),
        # Subscribing to an *AfterFinish channel makes the node deferred: it
        # runs only once nothing else is left to do in this invocation.
        "publish": NodeBuilder().subscribe_only("report").do(str.upper).write_to("published"),
        "archive": NodeBuilder()
        .subscribe_to("archive_gate", read=False)
        .read_from("count")
        .do(lambda inp: inp["count"])
        .write_to("archived"),
    }
    channels: dict[str, Any] = {
        "alerts": Topic(dict),
        "seen": Topic(str, accumulate=True),
        "count": BinaryOperatorAggregate(int, operator.add),
        "peak": MaxValue(),
        "status": AnyValue(str),
        "joined": NamedBarrierValue(str, {"dedupe", "score"}),
        "digest": LastValue(str),
        "tick": EphemeralValue(str),
        "report": LastValueAfterFinish(str),
        "published": LastValue(str),
        "archive_gate": NamedBarrierValueAfterFinish(str, {"dedupe", "score"}),
        "archived": LastValue(int),
        "debug_scratch": UntrackedValue(str),
    }
    return Pregel(
        nodes=nodes,  # type: ignore[arg-type]
        channels=channels,
        input_channels="alerts",
        output_channels=["digest", "published", "count", "archived", "status"],
        stream_channels=["digest", "status"],
        checkpointer=checkpointer,
        name="alert_pipeline",
    )
