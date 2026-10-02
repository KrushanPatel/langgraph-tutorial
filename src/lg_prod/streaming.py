"""Streaming: what a UI or API layer consumes while a graph runs.

LangGraph has three streaming surfaces:

1. **`stream(..., stream_mode=[...], version="v2")`**: the stable API. Every
   chunk is a typed `StreamPart` dict `{"type", "ns", "data", ...}`, so one
   `match` handles all modes. Modes: `values`, `updates`, `messages`
   (LLM tokens), `custom` (`get_stream_writer()`), `checkpoints`, `tasks`,
   `debug`. With `subgraphs=True`, `ns` names the subgraph each chunk came from.
2. **`stream_events(version="v3")`**: experimental. It returns a
   `GraphRunStream` whose *projections* (`run.values`, `run.messages`,
   `run.lifecycle`, `run.subgraphs`, plus any registered transformers) are
   single-consumer channels. Iterating them drives the graph; there is no
   background thread. **Subscribe to every projection you need before
   driving**: a `StreamChannel` drops items pushed before its subscriber exists.
3. **Graph callbacks**: `GraphCallbackHandler` in `config["callbacks"]`
   receives `GraphInterruptEvent` / `GraphResumeEvent`. That is the place
   for audit logs and notifications ("incident paused for approval").
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

from langchain_core.runnables import RunnableConfig
from langgraph.callbacks import (
    GraphCallbackHandler,
    GraphInterruptEvent,
    GraphLifecycleEvent,
    GraphLifecycleStatus,
    GraphResumeEvent,
)
from langgraph.pregel import Pregel
from langgraph.stream import ProtocolEvent, StreamChannel, StreamTransformer
from langgraph.types import (
    CheckpointPayload,
    CheckpointStreamPart,
    CheckpointTask,
    CustomStreamPart,
    DebugPayload,
    DebugStreamPart,
    MessagesStreamPart,
    StreamMode,
    StreamPart,
    TaskPayload,
    TaskResultPayload,
    TasksStreamPart,
    UpdatesStreamPart,
    ValuesStreamPart,
)

ALL_MODES: list[StreamMode] = [
    "values",
    "updates",
    "messages",
    "custom",
    "checkpoints",
    "tasks",
    "debug",
]


@dataclass
class StreamDigest:
    """What a UI would render, folded from v2 stream parts."""

    node_updates: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    custom_events: list[Any] = field(default_factory=list)
    checkpoints: list[CheckpointPayload[Any]] = field(default_factory=list)
    # Per-checkpoint view of pending tasks (name, error, interrupts).
    checkpoint_tasks: list[CheckpointTask] = field(default_factory=list)
    task_starts: list[TaskPayload] = field(default_factory=list)
    task_results: list[TaskResultPayload] = field(default_factory=list)
    debug_types: list[str] = field(default_factory=list)
    last_values: dict[str, Any] = field(default_factory=dict)
    namespaces: set[tuple[str, ...]] = field(default_factory=set)
    interrupts: list[Any] = field(default_factory=list)

    def add(self, part: StreamPart[Any, Any]) -> None:
        self.namespaces.add(part["ns"])
        match part["type"]:
            case "values":
                values: ValuesStreamPart[dict[str, Any]] = part
                self.last_values = values["data"]
                self.interrupts.extend(values["interrupts"])
            case "updates":
                updates: UpdatesStreamPart = part
                self.node_updates.extend(updates["data"])
            case "messages":
                messages: MessagesStreamPart = part
                chunk, _metadata = messages["data"]
                if chunk.text:
                    self.tokens.append(chunk.text)
            case "custom":
                custom: CustomStreamPart = part
                self.custom_events.append(custom["data"])
            case "checkpoints":
                cp: CheckpointStreamPart[dict[str, Any]] = part
                self.checkpoints.append(cp["data"])
                self.checkpoint_tasks.extend(cp["data"]["tasks"])
            case "tasks":
                task: TasksStreamPart = part
                data = task["data"]
                # Start events carry `triggers`; result events carry `result`.
                if "triggers" in data:
                    self.task_starts.append(cast(TaskPayload, data))
                else:
                    self.task_results.append(data)
            case "debug":
                debug: DebugStreamPart[dict[str, Any]] = part
                payload: DebugPayload[dict[str, Any]] = debug["data"]
                self.debug_types.append(payload["type"])


def iter_parts(
    graph: Pregel[Any, Any, Any, Any],
    payload: Any,
    config: RunnableConfig | None = None,
    *,
    modes: list[StreamMode] | None = None,
    subgraphs: bool = False,
    context: Any = None,
) -> Iterator[StreamPart[Any, Any]]:
    yield from graph.stream(
        payload,
        config,
        context=context,
        stream_mode=modes or ALL_MODES,
        subgraphs=subgraphs,
        version="v2",
    )


async def aiter_parts(
    graph: Pregel[Any, Any, Any, Any],
    payload: Any,
    config: RunnableConfig | None = None,
    *,
    modes: list[StreamMode] | None = None,
    subgraphs: bool = False,
    context: Any = None,
) -> AsyncIterator[StreamPart[Any, Any]]:
    async for part in graph.astream(
        payload,
        config,
        context=context,
        stream_mode=modes or ALL_MODES,
        subgraphs=subgraphs,
        version="v2",
    ):
        yield part


class CheckResultsTransformer(StreamTransformer):
    """v3 transformer: projects diagnostic `check` events into `run.checks`.

    `_native = True` exposes the projection as `run.checks`, in addition to
    `run.extensions["checks"]`. The named `StreamChannel` also re-emits each
    item as a protocol event on the main log under the method `checks`.
    """

    _native = True
    required_stream_modes: ClassVar[tuple[str, ...]] = ("custom",)

    def init(self) -> dict[str, Any]:
        self.checks: StreamChannel[dict[str, Any]] = StreamChannel("checks")
        return {"checks": self.checks}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] == "custom":
            data = event["params"]["data"]
            if isinstance(data, dict) and data.get("event") == "check":
                self.checks.push({"check": data["check"], "ok": data["ok"]})
        return True  # keep the event in the main log


class AuditCallbackHandler(GraphCallbackHandler):
    """Records graph lifecycle transitions, e.g. to page or to audit."""

    def __init__(self) -> None:
        self.events: list[GraphLifecycleEvent] = []

    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        self.events.append(event)

    def on_resume(self, event: GraphResumeEvent) -> None:
        self.events.append(event)

    def statuses(self) -> list[tuple[str, GraphLifecycleStatus, tuple[str, ...]]]:
        return [
            (
                "interrupt" if isinstance(e, GraphInterruptEvent) else "resume",
                e.status,
                tuple(ns.split(":")[0] for ns in e.checkpoint_ns),
            )
            for e in self.events
        ]
