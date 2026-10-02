"""The API boundary: how a web service drives the incident desk.

Responsibilities that belong here, not in graph nodes:

* **Thread ids.** One incident is one thread. The thread id goes in
  `config["configurable"]["thread_id"]`; identity goes in `context=`.
* **Interrupts as API states.** `ainvoke(..., version="v2")` returns a
  `GraphOutput` with `.value` and `.interrupts`. A non-empty `interrupts`
  becomes `status="awaiting_input"` with the questions to show a human.
  Answer with `Command(resume=value)` for one interrupt, or a map
  `{interrupt.id: value}` to answer several at once.
* **Graceful shutdown.** On SIGTERM, call `RunControl.request_drain()`. The
  run stops at the next superstep boundary, saves a checkpoint and raises
  `GraphDrained`. The service reports `status="drained"` and continues the
  run later with `ainvoke(None, ...)`.
* **Time travel.** `aget_state_history` lists every checkpoint. To fork,
  call `aupdate_state` with an old checkpoint's config (and `as_node`, so
  LangGraph knows which edges to follow) and continue from the new config.
* **Bulk import.** `abulk_update_state` writes several supersteps of
  `StateUpdate`s in one call, e.g. to migrate an incident from another tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphDrained
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import RunControl
from langgraph.types import Command, GraphOutput, Interrupt, PregelTask, StateSnapshot, StateUpdate

from lg_prod.config import Settings
from lg_prod.context import DeskContext
from lg_prod.graphs.triage import IncidentGraph


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    payload: dict[str, Any]
    response_schema: dict[str, Any] | None


@dataclass
class DeskReply:
    status: Literal["completed", "awaiting_input", "drained"]
    output: dict[str, Any] = field(default_factory=dict)
    questions: list[Question] = field(default_factory=list)


def _questions(interrupts: tuple[Interrupt, ...]) -> list[Question]:
    return [
        Question(
            id=i.id,
            kind=i.value.get("type", "unknown") if isinstance(i.value, dict) else "unknown",
            payload=i.value if isinstance(i.value, dict) else {"value": i.value},
            response_schema=i.response_schema,  # type: ignore[arg-type]
        )
        for i in interrupts
    ]


class IncidentDeskService:
    def __init__(self, graph: IncidentGraph, settings: Settings) -> None:
        self._graph = graph
        self._settings = settings

    def _config(self, thread_id: str) -> RunnableConfig:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._settings.recursion_limit,
            "tags": ["incident-desk"],
            "metadata": {"thread_id": thread_id},
        }

    async def _run(
        self,
        payload: Any,
        config: RunnableConfig,
        ctx: DeskContext,
        control: RunControl | None,
    ) -> DeskReply:
        try:
            out = await self._graph.ainvoke(
                payload,
                config,
                context=ctx,
                durability=self._settings.durability,
                control=control,
                version="v2",
            )
        except GraphDrained:
            return DeskReply(status="drained")
        assert isinstance(out, GraphOutput)
        if out.interrupts:
            return DeskReply("awaiting_input", dict(out.value or {}), _questions(out.interrupts))
        return DeskReply("completed", dict(out.value))

    # -------------------------------------------------------------- commands

    async def report(
        self,
        thread_id: str,
        report: str,
        service: str,
        ctx: DeskContext,
        *,
        control: RunControl | None = None,
    ) -> DeskReply:
        return await self._run(
            {"report": report, "service": service}, self._config(thread_id), ctx, control
        )

    async def answer(
        self, thread_id: str, ctx: DeskContext, answers: dict[str, Any] | Any
    ) -> DeskReply:
        """Resume. `answers` is either one value or `{question_id: value}`."""
        return await self._run(Command(resume=answers), self._config(thread_id), ctx, None)

    async def continue_run(
        self, thread_id: str, ctx: DeskContext, *, control: RunControl | None = None
    ) -> DeskReply:
        """Continue after a drain or a static breakpoint (input `None`)."""
        return await self._run(None, self._config(thread_id), ctx, control)

    # --------------------------------------------------------------- queries

    async def snapshot(self, thread_id: str) -> StateSnapshot:
        return await self._graph.aget_state(self._config(thread_id), subgraphs=True)

    async def pending_tasks(self, thread_id: str) -> list[PregelTask]:
        return list((await self.snapshot(thread_id)).tasks)

    async def history(self, thread_id: str, *, limit: int | None = None) -> list[StateSnapshot]:
        return [
            s async for s in self._graph.aget_state_history(self._config(thread_id), limit=limit)
        ]

    # ----------------------------------------------------------- time travel

    async def fork(
        self,
        thread_id: str,
        checkpoint: StateSnapshot,
        patch: dict[str, Any],
        *,
        as_node: str,
    ) -> RunnableConfig:
        """Create a new branch from an old checkpoint with `patch` applied."""
        return await self._graph.aupdate_state(checkpoint.config, patch, as_node=as_node)

    async def resume_from(self, config: RunnableConfig, ctx: DeskContext) -> DeskReply:
        """Run forward from a specific checkpoint (replay or a fork)."""
        return await self._run(None, config, ctx, None)

    async def import_incident(
        self, thread_id: str, supersteps: list[list[StateUpdate]]
    ) -> RunnableConfig:
        return await self._graph.abulk_update_state(self._config(thread_id), supersteps)

    async def clear_conversation(self, thread_id: str) -> None:
        """Drop every message in the thread (e.g. GDPR erasure of free text)."""
        await self._graph.aupdate_state(
            self._config(thread_id), {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
        )
