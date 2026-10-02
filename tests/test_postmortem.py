"""Functional API: tasks, futures, previous/final, interrupts, policies."""

from __future__ import annotations

import asyncio

import pytest
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeTimeoutError
from langgraph.pregel import Pregel
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from lg_prod.graphs.postmortem import SECTIONS, TemplateDrafter, build_postmortem_workflow
from tests.conftest import thread

REQUEST = {"incident_id": "INC-9", "service": "checkout", "timeline": ["db failing", "failover"]}


async def _run_to_review(wf: Pregel, cfg: dict, lead) -> dict:  # type: ignore[no-untyped-def, type-arg]
    out = await wf.ainvoke(REQUEST, cfg, context=lead, version="v2")
    [review] = out.interrupts
    return review.value  # type: ignore[no-any-return]


async def test_review_interrupt_then_final_report(lead):  # type: ignore[no-untyped-def]
    drafter = TemplateDrafter()
    store = InMemoryStore()
    wf = build_postmortem_workflow(drafter, checkpointer=InMemorySaver(), store=store)
    cfg = thread("pm1")

    review = await _run_to_review(wf, cfg, lead)
    assert review["revision"] == 1
    assert set(review["report"]["sections"]) == set(SECTIONS)

    report = await wf.ainvoke(
        Command(resume={"edits": {"impact": "12% of checkouts failed"}}), cfg, context=lead
    )

    assert report["sections"]["impact"] == "12% of checkouts failed"
    assert report["approved_by"] == "u-lead"
    assert store.get(("postmortems",), "INC-9") is not None
    # Resuming re-ran the entrypoint body, but finished tasks were not re-run.
    assert sorted(drafter.calls) == sorted(SECTIONS)


async def test_previous_holds_saved_value_not_return_value(lead):  # type: ignore[no-untyped-def]
    wf = build_postmortem_workflow(TemplateDrafter(), checkpointer=InMemorySaver())
    cfg = thread("pm2")
    await _run_to_review(wf, cfg, lead)
    await wf.ainvoke(Command(resume={"edits": {}}), cfg, context=lead)

    review = await _run_to_review(wf, cfg, lead)  # same thread, second run

    assert review["revision"] == 2  # `previous` was the saved int, 1
    # `get_state().values` shows the last *returned* value (the full report);
    # only `previous` sees what `entrypoint.final(save=...)` stored.
    assert wf.get_state(cfg).values["sections"]


async def test_task_retry_policy(lead):  # type: ignore[no-untyped-def]
    wf = build_postmortem_workflow(TemplateDrafter(), checkpointer=InMemorySaver(), flaky_lookups=2)

    review = await _run_to_review(wf, thread("pm3"), lead)

    assert "deploy history" in review["report"]["sections"]["summary"]


async def test_task_cache_across_threads(lead):  # type: ignore[no-untyped-def]
    drafter = TemplateDrafter()
    wf = build_postmortem_workflow(drafter, checkpointer=InMemorySaver(), cache=InMemoryCache())

    await _run_to_review(wf, thread("c1"), lead)
    await _run_to_review(wf, thread("c2"), lead)

    assert len(drafter.calls) == len(SECTIONS)  # second thread hit the cache


async def test_custom_stream_from_tasks(lead):  # type: ignore[no-untyped-def]
    wf = build_postmortem_workflow(TemplateDrafter(), checkpointer=InMemorySaver())

    events = [
        chunk
        async for chunk in wf.astream(REQUEST, thread("s1"), context=lead, stream_mode="custom")
    ]

    assert sorted(e["drafting"] for e in events) == sorted(SECTIONS)


async def test_heartbeat_keeps_idle_timeout_alive(lead):  # type: ignore[no-untyped-def]
    # 4 sections x 0.15s = 0.6s total, more than the 0.3s idle timeout,
    # but there is never 0.3s without a heartbeat.
    wf = build_postmortem_workflow(
        TemplateDrafter(), checkpointer=InMemorySaver(), index_timeout_s=0.3
    )
    cfg = thread("hb")
    await _run_to_review(wf, cfg, lead)
    report = await wf.ainvoke(Command(resume={"edits": {}}), cfg, context=lead)
    assert report["revision"] == 1


async def test_idle_timeout_fires_without_heartbeats(lead):  # type: ignore[no-untyped-def]
    from langgraph.func import entrypoint, task
    from langgraph.types import TimeoutPolicy

    @task(timeout=TimeoutPolicy(idle_timeout=0.05, refresh_on="heartbeat"))
    async def stuck() -> None:
        await asyncio.sleep(1)

    @entrypoint()
    async def wf(_: int) -> None:
        await stuck()

    with pytest.raises(NodeTimeoutError):
        await wf.ainvoke(1)
