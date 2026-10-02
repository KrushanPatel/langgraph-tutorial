"""Streaming: v2 typed parts for every mode, v3 run streams, graph callbacks."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolCallTransformer
from langgraph.stream import (
    AsyncGraphRunStream,
    AsyncSubgraphRunStream,
    CheckpointsTransformer,
    CustomTransformer,
    DebugTransformer,
    GraphRunStream,
    LifecyclePayload,
    LifecycleTransformer,
    SubgraphRunStream,
    SubgraphStatus,
    SubgraphTransformer,
    TasksTransformer,
    UpdatesTransformer,
)
from langgraph.types import Command

from lg_prod.config import Settings
from lg_prod.graphs.runbook_agent import build_runbook_agent
from lg_prod.graphs.triage import build_incident_graph
from lg_prod.services import FakeDiagnostics, demo_services
from lg_prod.streaming import (
    AuditCallbackHandler,
    CheckResultsTransformer,
    StreamDigest,
    aiter_parts,
    iter_parts,
)
from tests.conftest import call, script, thread

SEARCH_CPU = {"report": "search cpu saturation", "service": "search"}
AUTH_DOWN = {"report": "auth is down", "service": "auth"}

pytestmark = pytest.mark.filterwarnings("ignore::langchain_core._api.LangChainBetaWarning")


def _escalating_desk():  # type: ignore[no-untyped-def]
    services = demo_services()
    services.diagnostics = FakeDiagnostics(broken={("auth", "dependencies")})
    return build_incident_graph(Settings(_env_file=None), services, checkpointer=InMemorySaver())  # type: ignore[call-arg]


async def test_every_stream_mode_as_typed_v2_parts(desk, lead):  # type: ignore[no-untyped-def]
    digest = StreamDigest()
    async for part in aiter_parts(desk, SEARCH_CPU, thread("s1"), context=lead):
        digest.add(part)

    assert digest.node_updates[:3] == ["intake", "classify", "fetch_metrics"]
    assert {"event": "intake", "incident_id": "INC-s1"} in digest.custom_events
    assert any(e.get("event") == "check" for e in digest.custom_events)
    # `messages` streams LLM tokens AND messages nodes write to state
    # (intake's HumanMessage, close's push_message summary).
    assert digest.tokens == [SEARCH_CPU["report"], "INC-s1 remediated: 1 failing check(s): cpu"]
    assert digest.last_values["status"] == "remediated"  # output schema applied
    assert {t["name"] for t in digest.task_starts} >= {"intake", "run_check", "assess"}
    assert all(r["error"] is None for r in digest.task_results)
    assert digest.checkpoints[-1]["next"] == []
    assert "assess" in {t["name"] for t in digest.checkpoint_tasks}
    assert set(digest.debug_types) == {"checkpoint", "task", "task_result"}
    assert () in digest.namespaces


async def test_subgraph_namespaces_and_interrupts_in_stream(lead):  # type: ignore[no-untyped-def]
    desk = _escalating_desk()
    digest = StreamDigest()
    async for part in aiter_parts(
        desk, AUTH_DOWN, thread("s2"), modes=["updates", "values"], subgraphs=True, context=lead
    ):
        digest.add(part)

    assert any(ns and ns[0].startswith("escalate:") for ns in digest.namespaces)
    assert digest.interrupts[0].value["type"] == "page_ack"


def test_tag_nostream_hides_internal_llm_calls():
    internal = script("internal reasoning").with_config(tags=[TAG_NOSTREAM])
    public = script("hello user")

    def node(state: MessagesState) -> dict:  # type: ignore[type-arg]
        internal.invoke(state["messages"])
        return {"messages": [public.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("reply", node)
    builder.add_edge(START, "reply")
    graph = builder.compile()

    tokens = [
        part["data"][0].text
        for part in iter_parts(graph, {"messages": [HumanMessage("hi")]}, modes=["messages"])
    ]
    assert "".join(tokens) == "hello user"


def test_v3_run_stream_with_tool_call_and_custom_transformers(services, lead):  # type: ignore[no-untyped-def]
    model = script(call("restart_service", {"reason": "oom"}), "Restarted.")
    agent = build_runbook_agent(model, services)

    run = agent.stream_events(
        {"messages": [HumanMessage("restart")], "service": "search", "notes": []},
        version="v3",
        context=lead,
        transformers=[UpdatesTransformer, CustomTransformer, ToolCallTransformer],
    )
    assert isinstance(run, GraphRunStream)
    # Subscribe to every projection BEFORE pulling: channels drop items that
    # arrive before their subscriber exists.
    custom, updates = iter(run.custom), iter(run.updates)
    seen = []
    for tool_call in run.tool_calls:  # this loop drives the run
        # Read nested streams inside the loop, while the run is still live.
        seen.append((tool_call.tool_name, list(tool_call.output_deltas), tool_call))

    [(name, deltas, restart)] = seen
    assert name == "restart_service"
    assert deltas == [{"progress": 0}, {"progress": 50}, {"progress": 100}]
    assert restart.completed and restart.error is None
    assert list(custom) == [{"event": "restart", "service": "search"}]
    assert [next(iter(u)) for u in updates] == ["agent", "tools", "agent"]
    assert run.interrupted is False
    # v3 is content-block-centric: final message content is a list of blocks.
    assert run.output["messages"][-1].content == [
        {"type": "text", "text": "Restarted.", "index": 0}
    ]


def test_v3_projection_subscribed_late_misses_items(services, lead):  # type: ignore[no-untyped-def]
    model = script(call("restart_service", {"reason": "oom"}), "Restarted.")
    agent = build_runbook_agent(model, services)
    run = agent.stream_events(
        {"messages": [HumanMessage("x")], "service": "search", "notes": []},
        version="v3",
        context=lead,
        transformers=[CustomTransformer],
    )
    list(run.values)  # drives the whole run first
    assert list(run.custom) == []  # subscribed too late


def test_v3_builtin_transformers_compiled_into_the_graph(lead):  # type: ignore[no-untyped-def]
    from langgraph.pregel import Pregel

    from lg_prod.graphs.alert_pipeline import build_alert_pipeline

    base = build_alert_pipeline(InMemorySaver())
    pipeline = Pregel(  # same graph, with compile-time stream transformers
        nodes=base.nodes,
        channels=base.channels,
        input_channels=base.input_channels,
        output_channels=base.output_channels,
        checkpointer=base.checkpointer,
        stream_transformers=[CheckpointsTransformer, TasksTransformer, DebugTransformer],
    )
    run = pipeline.stream_events(
        [{"id": "a", "source": "s", "level": "info"}], thread("v3"), version="v3"
    )
    checkpoints, tasks, debug = iter(run.checkpoints), iter(run.tasks), iter(run.debug)
    output = run.output

    assert output["count"] == 1
    assert len(list(checkpoints)) >= 3
    assert {t["name"] for t in tasks if "triggers" in t} >= {"dedupe", "score"}
    assert list(debug)


async def test_async_v3_subgraphs_lifecycle_and_custom_projection(lead):  # type: ignore[no-untyped-def]
    desk = _escalating_desk()
    run = await desk.astream_events(
        AUTH_DOWN,
        thread("v3a"),
        version="v3",
        context=lead,
        transformers=[CheckResultsTransformer],  # lifecycle/subgraphs are built in
    )
    assert isinstance(run, AsyncGraphRunStream)

    async def drain_subgraphs() -> list[str]:
        names = []
        async for sub in run.subgraphs:
            assert isinstance(sub, AsyncSubgraphRunStream)
            names.append(sub.path[0].split(":")[0])
            async for _ in sub.values:
                pass
        return names

    async def drain_lifecycle() -> list[LifecyclePayload]:
        return [event async for event in run.lifecycle]

    async def drain_checks() -> list[dict]:  # type: ignore[type-arg]
        return [c async for c in run.checks]

    subgraphs, lifecycle, checks = await asyncio.gather(
        drain_subgraphs(), drain_lifecycle(), drain_checks()
    )

    assert subgraphs == ["escalate"]
    statuses: list[SubgraphStatus] = [e["event"] for e in lifecycle]  # type: ignore[misc]
    assert statuses == ["started", "interrupted"]
    assert {c["check"] for c in checks} == {"db", "deploys", "dependencies"}
    assert await run.interrupted()
    [ack] = await run.interrupts()
    assert ack.value["type"] == "page_ack"


async def test_graph_callbacks_see_interrupt_and_resume(lead):  # type: ignore[no-untyped-def]
    desk = _escalating_desk()
    audit = AuditCallbackHandler()
    cfg = {**thread("cb"), "callbacks": [audit]}

    await desk.ainvoke(AUTH_DOWN, cfg, context=lead)
    await desk.ainvoke(Command(resume={"acknowledged_by": "u-lead"}), cfg, context=lead)

    assert audit.statuses() == [
        ("interrupt", "pending", ()),
        ("resume", "pending", ()),
        ("resume", "pending", ("escalate",)),
    ]


@pytest.mark.parametrize("builtin", [LifecycleTransformer, SubgraphTransformer])
def test_builtin_transformers_are_always_registered(builtin, lead):  # type: ignore[no-untyped-def]
    desk = _escalating_desk()
    with pytest.raises(ValueError, match="conflict with already-registered keys"):
        desk.stream_events(AUTH_DOWN, thread("dup"), version="v3", transformers=[builtin])


def test_sync_subgraph_streams(lead):  # type: ignore[no-untyped-def]
    from typing import TypedDict

    from lg_prod.graphs.escalation import build_escalation_graph

    class Outer(TypedDict, total=False):
        service: str
        severity: str
        summary: str
        status: str

    # A parent whose only node is the escalation subgraph (sev3: no interrupt).
    builder = StateGraph(Outer)
    builder.add_node("escalate", build_escalation_graph(demo_services()))
    builder.add_node("close", lambda s: {"status": "closed"})
    builder.add_edge(START, "escalate")
    graph = builder.compile()

    run = graph.stream_events(
        {"service": "auth", "severity": "sev3", "summary": "x"}, version="v3", context=lead
    )
    names = []
    for sub in run.subgraphs:
        assert isinstance(sub, SubgraphRunStream)
        names.append(sub.path[0].split(":")[0])
        list(sub.values)
    assert names == ["escalate"]
