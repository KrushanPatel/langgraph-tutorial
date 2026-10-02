"""The incident desk graph: routing, HITL, subgraphs, policies, special channels."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, GraphOutput, Interrupt, StateSnapshot
from pydantic import ValidationError

from lg_prod.graphs.triage import build_incident_graph
from lg_prod.models import Classification, RuleClassifier
from lg_prod.services import FakeDiagnostics, FakeMetrics
from tests.conftest import thread

CHECKOUT = {"report": "checkout returning 500 errors for customers", "service": "checkout"}
SEARCH_CPU = {"report": "search cpu saturation", "service": "search"}
AUTH_DOWN = {"report": "auth is down, total outage", "service": "auth"}


@dataclass
class CountingClassifier:
    calls: int = 0
    fail: bool = False
    inner: RuleClassifier = field(default_factory=RuleClassifier)

    def classify(self, report: str, service: str) -> Classification:
        self.calls += 1
        if self.fail:
            raise RuntimeError("LLM provider down")
        return self.inner.classify(report, service)


async def test_mild_incident_is_auto_remediated_by_policy(desk, lead, services):  # type: ignore[no-untyped-def]
    out = await desk.ainvoke(SEARCH_CPU, thread("t1"), context=lead, version="v2")

    assert isinstance(out, GraphOutput)
    assert out.interrupts == ()
    assert out.value["severity"] == "sev3"
    assert out.value["status"] == "remediated"
    assert services.remediator.applied == [("search", "scale_out")]  # type: ignore[attr-defined]


async def test_output_schema_hides_internal_and_private_keys(desk, lead):  # type: ignore[no-untyped-def]
    out = await desk.ainvoke(SEARCH_CPU, thread("t1"), context=lead)

    assert set(out) <= {"incident_id", "severity", "status", "summary", "actions_taken"}
    state = (await desk.aget_state(thread("t1"))).values
    assert "findings" in state and "timeline" in state  # internal: in state, not output
    assert "runbook" not in out and "similar_incidents" not in out  # private


async def test_severe_incident_pauses_for_approval_then_resumes(desk, lead, services):  # type: ignore[no-untyped-def]
    cfg = thread("t2")
    out = await desk.ainvoke(CHECKOUT, cfg, context=lead, version="v2")

    [pending] = out.interrupts
    assert isinstance(pending, Interrupt)
    assert pending.value == {
        "type": "approval",
        "incident_id": "INC-t2",
        "action": "failover_db",
        "severity": "sev2",
    }
    # The JSON schema of the expected answer travels with the interrupt.
    assert pending.response_schema is not None
    assert pending.response_schema["required"] == ["approved", "note"]
    snapshot = await desk.aget_state(cfg)
    assert isinstance(snapshot, StateSnapshot)
    assert snapshot.next == ("approve",)
    assert services.remediator.applied == []  # type: ignore[attr-defined]

    ui = (await desk.aget_state(cfg)).values["ui"]
    assert [m["name"] for m in ui] == ["incident_card", "approval_request"]

    out = await desk.ainvoke(
        Command(resume={"approved": True, "note": "go"}), cfg, context=lead, version="v2"
    )

    assert out.value["status"] == "remediated"
    values = (await desk.aget_state(cfg)).values
    assert values["approved_by"] == "u-lead"
    # `delete_ui_message` in `close` removed the now-obsolete approval card.
    assert [m["name"] for m in values["ui"]] == ["incident_card"]


async def test_resume_value_is_validated_against_response_schema(desk, lead):  # type: ignore[no-untyped-def]
    cfg = thread("t3")
    await desk.ainvoke(CHECKOUT, cfg, context=lead)

    with pytest.raises(ValidationError):
        await desk.ainvoke(Command(resume={"approved": "maybe?"}), cfg, context=lead)


async def test_approval_from_engineer_is_not_honoured_and_escalates(desk, lead, engineer, services):  # type: ignore[no-untyped-def]
    cfg = thread("t4")
    await desk.ainvoke(CHECKOUT, cfg, context=lead)

    # Authorization is checked against the context of the *resuming* call.
    out = await desk.ainvoke(
        Command(resume={"approved": True, "note": "lgtm"}), cfg, context=engineer, version="v2"
    )

    assert out.value["status"] == "escalated"
    assert services.remediator.applied == []  # type: ignore[attr-defined]
    assert services.pager.pages[0][0] == "platform"  # type: ignore[attr-defined]


async def test_sev1_escalation_interrupt_inside_subgraph(settings, lead):  # type: ignore[no-untyped-def]
    from lg_prod.services import demo_services

    services = demo_services()
    services.diagnostics = FakeDiagnostics(broken={("auth", "dependencies")})  # no runbook
    desk = build_incident_graph(settings, services, checkpointer=InMemorySaver())
    cfg = thread("t5")

    out = await desk.ainvoke(AUTH_DOWN, cfg, context=lead, version="v2")

    [ack] = out.interrupts
    assert ack.value == {"type": "page_ack", "page_id": "PAGE-1"}
    snapshot = await desk.aget_state(cfg, subgraphs=True)
    [task] = snapshot.tasks
    assert task.name == "escalate"
    assert isinstance(task.state, StateSnapshot)  # the subgraph's own state
    assert task.state.next == ("wait_for_ack",)

    out = await desk.ainvoke(
        Command(resume={"acknowledged_by": "u-lead"}), cfg, context=lead, version="v2"
    )

    # `Command.PARENT` from the subgraph jumped to the parent's `close` node.
    assert out.value["status"] == "escalated"
    assert out.value["actions_taken"] == ["escalated via PAGE-1"]


async def test_classifier_failure_routes_through_error_handler(settings, services, lead):  # type: ignore[no-untyped-def]
    classifier = CountingClassifier(fail=True)
    desk = build_incident_graph(settings, services, classifier=classifier)

    out = await desk.ainvoke(SEARCH_CPU, context=lead)

    # set_node_defaults gave `classify` a RetryPolicy for TransientBackendError
    # only, so a RuntimeError goes straight to the handler: 1 call.
    assert classifier.calls == 1
    assert out["severity"] == "sev3"
    assert out["status"] == "remediated"


async def test_cache_policy_needs_a_cache_at_compile_time(settings, services, lead):  # type: ignore[no-untyped-def]
    cached = CountingClassifier()
    desk = build_incident_graph(settings, services, classifier=cached, cache=InMemoryCache())
    await desk.ainvoke(SEARCH_CPU, context=lead)
    await desk.ainvoke(SEARCH_CPU, context=lead)
    assert cached.calls == 1

    await desk.aclear_cache()  # cache administration lives on the compiled graph
    await desk.ainvoke(SEARCH_CPU, context=lead)
    assert cached.calls == 2

    uncached = CountingClassifier()
    desk = build_incident_graph(settings, services, classifier=uncached)  # no cache=
    await desk.ainvoke(SEARCH_CPU, context=lead)
    await desk.ainvoke(SEARCH_CPU, context=lead)
    assert uncached.calls == 2  # CachePolicy silently does nothing


async def test_transient_metrics_errors_are_retried(settings, services, lead):  # type: ignore[no-untyped-def]
    services.metrics = FakeMetrics(data={"search": {"cpu": 0.9}}, failures_before_success=2)
    desk = build_incident_graph(settings, services)

    out = await desk.ainvoke(SEARCH_CPU, context=lead)

    assert services.metrics.calls == 3
    assert out["status"] == "remediated"


async def test_metrics_timeout_degrades_via_error_handler(settings, services, lead):  # type: ignore[no-untyped-def]
    services.metrics = FakeMetrics(data={"search": {"cpu": 0.9}}, delay_s=5)
    desk = build_incident_graph(settings.model_copy(update={"node_max_attempts": 1}), services)

    out = await desk.ainvoke(SEARCH_CPU, thread("t6"), context=lead)

    assert out["status"] == "remediated"  # diagnostics still ran


async def test_permanent_errors_are_not_retried(settings, services, lead):  # type: ignore[no-untyped-def]
    services.metrics = FakeMetrics(data={})  # unknown service -> PermanentBackendError
    desk = build_incident_graph(settings, services)

    await desk.ainvoke(SEARCH_CPU, context=lead)

    assert services.metrics.calls == 1  # first matching RetryPolicy: max_attempts=1


def test_timeouts_require_async_execution(desk, lead, services):  # type: ignore[no-untyped-def]
    # Sync `invoke` cannot enforce a node timeout, so `fetch_metrics` raises
    # ValueError on every sync run - and its error_handler silently degrades.
    out = desk.invoke(SEARCH_CPU, thread("t7"), context=lead)

    assert out["status"] == "remediated"
    assert services.metrics.calls == 0  # metrics were never fetched
    timeline = desk.get_state(thread("t7")).values["timeline"]
    assert any("only supported for async nodes" in line for line in timeline)


def test_sync_node_with_timeout_is_rejected_at_compile_time():
    from langgraph.graph import START, StateGraph

    builder = StateGraph(dict)
    builder.add_node("n", lambda s: s, timeout=1.0)
    builder.add_edge(START, "n")
    with pytest.raises(ValueError, match="only supported for async nodes"):
        builder.compile()


async def test_flaky_probe_degrades_on_final_attempt(settings, services, lead):  # type: ignore[no-untyped-def]
    services.diagnostics = FakeDiagnostics(crashing={"disk"})
    desk = build_incident_graph(settings, services, checkpointer=InMemorySaver())

    await desk.ainvoke(SEARCH_CPU, thread("t8"), context=lead)

    findings = {f["check"]: f for f in (await desk.aget_state(thread("t8"))).values["findings"]}
    assert findings["disk"]["ok"] is False
    assert "probe error" in findings["disk"]["detail"]
    assert services.diagnostics.calls.count(("search", "disk")) == settings.node_max_attempts


async def test_special_channels(desk, lead):  # type: ignore[no-untyped-def]
    cfg = thread("t9")
    await desk.ainvoke(SEARCH_CPU, cfg, context=lead)
    values = (await desk.aget_state(cfg)).values

    assert "trace_url" not in values  # UntrackedValue: never checkpointed
    assert "handoff_note" not in values  # EphemeralValue: cleared after one step
    assert values["audit"] == ["intake:u-lead", "close", "remember"]  # DeltaChannel
    assert values["timeline"][-1] == "closed as remediated"  # Overwrite-compacted
    [card] = values["ui"]  # two push_ui_message calls, merged by id
    assert card["name"] == "incident_card"
    assert card["props"] == {
        "incident_id": "INC-t9",
        "service": "search",
        "status": "remediated",
    }
    assert values["messages"][-1].content.startswith("INC-t9 remediated")  # push_message


async def test_long_term_memory_is_recalled_on_next_incident(desk, lead, store):  # type: ignore[no-untyped-def]
    await desk.ainvoke(SEARCH_CPU, thread("a"), context=lead)
    [item] = store.search(lead.memory_namespace)
    assert item.key == "INC-a"

    await desk.ainvoke(SEARCH_CPU, thread("b"), context=lead)

    timeline = (await desk.aget_state(thread("b"))).values["timeline"]
    assert any("seen before: search:" in line for line in timeline)

    # Memory is filtered by service: a checkout incident doesn't recall search.
    await desk.ainvoke(
        {"report": "checkout cpu saturation", "service": "checkout"}, thread("c"), context=lead
    )
    timeline = (await desk.aget_state(thread("c"))).values["timeline"]
    assert not any("seen before" in line for line in timeline)


async def test_static_breakpoint_before_remediation(settings, services, lead):  # type: ignore[no-untyped-def]
    desk = build_incident_graph(
        settings, services, checkpointer=InMemorySaver(), interrupt_before=["remediate"]
    )
    cfg = thread("bp")

    await desk.ainvoke(SEARCH_CPU, cfg, context=lead)
    assert (await desk.aget_state(cfg)).next == ("remediate",)
    assert services.remediator.applied == []  # type: ignore[attr-defined]

    out = await desk.ainvoke(None, cfg, context=lead)  # `None` = continue

    assert out["status"] == "remediated"


async def test_step_budget_escalates_instead_of_crashing(desk, lead):  # type: ignore[no-untyped-def]
    with pytest.raises(GraphRecursionError):
        await desk.ainvoke(SEARCH_CPU, {**thread("r1"), "recursion_limit": 4}, context=lead)

    # With a tight but sufficient budget, `assess` sees RemainingSteps < 6 and
    # escalates (2 more steps) instead of remediating (4 more steps).
    out = await desk.ainvoke(SEARCH_CPU, {**thread("r2"), "recursion_limit": 10}, context=lead)
    assert out["status"] == "escalated"


def test_graph_drawing_includes_subgraph_with_xray(desk):  # type: ignore[no-untyped-def]
    assert isinstance(desk, CompiledStateGraph)
    flat = desk.get_graph()
    deep = desk.get_graph(xray=True)

    assert "escalate" in flat.nodes
    assert "escalate:page_team" in deep.nodes
    mermaid = desk.get_graph().draw_mermaid()
    assert "assess" in mermaid and "-.->" in mermaid  # conditional edges are dotted
