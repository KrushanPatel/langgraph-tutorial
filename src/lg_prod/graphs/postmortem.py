"""Post-mortem writer built with the Functional API (`@entrypoint` + `@task`).

The Functional API trades the explicit graph for plain Python control flow,
while keeping LangGraph's persistence, interrupts, streaming and policies.
Rules that matter in production:

* **Only `@task` results are checkpointed.** On resume the entrypoint body
  re-runs from the top, and finished tasks return their saved result instead
  of executing again. Side effects (API calls, randomness, time) must live
  in tasks, or they repeat on every resume.
* **Tasks return futures.** Call several, then `.result()` them, and they run
  in parallel.
* **`previous`** is the value the entrypoint saved on the last run of the
  same thread. `entrypoint.final(value=..., save=...)` returns one thing to
  the caller while saving something else for next time (here: a revision
  counter).
* **Policies:** `retry_policy`, `cache_policy` (needs `cache=`) and
  `timeout` work on tasks exactly as on graph nodes. `TimeoutPolicy(
  idle_timeout=..., refresh_on="heartbeat")` kills a task only if it stops
  calling `runtime.heartbeat()`, the right choice for long jobs that make
  steady progress.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, TypedDict

from langgraph.cache.base import BaseCache
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.func import entrypoint, task
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime, get_runtime
from langgraph.store.base import BaseStore
from langgraph.types import CachePolicy, RetryPolicy, TimeoutPolicy, interrupt

from lg_prod.context import DeskContext
from lg_prod.services import TransientBackendError

SECTIONS = ("summary", "impact", "root_cause", "follow_ups")


class PostmortemRequest(TypedDict):
    incident_id: str
    service: str
    timeline: list[str]


class Report(TypedDict):
    incident_id: str
    revision: int
    sections: dict[str, str]
    approved_by: str | None


class Drafter(Protocol):
    def draft(self, section: str, facts: list[str]) -> str: ...


class TemplateDrafter:
    """Deterministic stand-in for an LLM drafter."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def draft(self, section: str, facts: list[str]) -> str:
        self.calls.append(section)
        return f"{section}: " + "; ".join(facts[-3:])


def build_postmortem_workflow(
    drafter: Drafter,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    store: BaseStore | None = None,
    cache: BaseCache[Any] | None = None,
    flaky_lookups: int = 0,
    index_timeout_s: float = 1.0,
) -> Pregel[Any, Any, Any, Any]:
    lookups = {"failures_left": flaky_lookups}

    @task(
        retry_policy=RetryPolicy(
            max_attempts=3, initial_interval=0.01, jitter=False, retry_on=TransientBackendError
        )
    )
    def enrich_timeline(timeline: list[str]) -> list[str]:
        if lookups["failures_left"] > 0:
            lookups["failures_left"] -= 1
            raise TransientBackendError("deploy API flaked")
        return [*timeline, "deploy history: no deploy in the last hour"]

    @task(cache_policy=CachePolicy(ttl=3600))
    def draft_section(section: str, facts: list[str]) -> str:
        get_stream_writer()({"drafting": section})
        return drafter.draft(section, facts)

    @task(timeout=TimeoutPolicy(idle_timeout=index_timeout_s, refresh_on="heartbeat"))
    async def index_for_search(report: Report) -> int:
        # Long job with steady progress: each heartbeat resets the idle timer,
        # so the total runtime may exceed `index_timeout_s` safely.
        runtime = get_runtime()
        for _ in report["sections"]:
            await asyncio.sleep(index_timeout_s / 2)
            runtime.heartbeat()
        return len(report["sections"])

    @entrypoint(
        checkpointer=checkpointer,
        store=store,
        cache=cache,
        context_schema=DeskContext,
    )
    async def postmortem(
        request: PostmortemRequest,
        *,
        previous: int | None,
        runtime: Runtime[DeskContext],
    ) -> entrypoint.final[Report, int]:
        facts = await enrich_timeline(request["timeline"])
        # Fan out: create every future first, then await them together.
        futures = [draft_section(name, facts) for name in SECTIONS]
        sections = dict(zip(SECTIONS, await asyncio.gather(*futures), strict=True))
        revision = (previous or 0) + 1
        report: Report = {
            "incident_id": request["incident_id"],
            "revision": revision,
            "sections": sections,
            "approved_by": None,
        }
        review = interrupt({"type": "review", "revision": revision, "report": report})
        if review.get("edits"):
            report["sections"] = {**sections, **review["edits"]}
        report["approved_by"] = runtime.context.user_id
        await index_for_search(report)
        if runtime.store is not None:
            runtime.store.put(("postmortems",), request["incident_id"], dict(report))
        # Caller gets the full report; the thread only remembers the revision.
        return entrypoint.final(value=report, save=revision)

    return postmortem
