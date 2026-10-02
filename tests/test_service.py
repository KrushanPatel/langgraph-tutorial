"""Service boundary: API statuses, drain/resume, time travel, bulk import."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.runtime import RunControl
from langgraph.types import PregelTask, StateSnapshot, StateUpdate

from lg_prod.graphs.triage import build_incident_graph
from lg_prod.service import IncidentDeskService
from lg_prod.services import FakeMetrics


def _service(desk, settings) -> IncidentDeskService:  # type: ignore[no-untyped-def]
    return IncidentDeskService(desk, settings)


async def test_report_and_answer_round_trip(desk, settings, lead):  # type: ignore[no-untyped-def]
    svc = _service(desk, settings)

    reply = await svc.report("i1", "checkout returning 500 errors for customers", "checkout", lead)

    assert reply.status == "awaiting_input"
    [question] = reply.questions
    assert question.kind == "approval"
    assert question.response_schema is not None
    [task] = await svc.pending_tasks("i1")
    assert isinstance(task, PregelTask) and task.name == "approve"
    assert task.interrupts[0].id == question.id

    # Answer by interrupt id: the form that also works for several questions.
    reply = await svc.answer("i1", lead, {question.id: {"approved": True, "note": "ok"}})

    assert reply.status == "completed"
    assert reply.output["status"] == "remediated"


@dataclass
class DrainingMetrics:
    """Simulates SIGTERM arriving while metrics are being fetched."""

    control: RunControl
    inner: FakeMetrics

    async def fetch(self, service: str) -> dict[str, float]:
        self.control.request_drain("SIGTERM")
        return await self.inner.fetch(service)


async def test_drain_stops_at_superstep_boundary_and_resumes(settings, services, lead):  # type: ignore[no-untyped-def]
    from langgraph.checkpoint.memory import InMemorySaver

    control = RunControl()
    services.metrics = DrainingMetrics(control, FakeMetrics(data={"search": {"cpu": 0.9}}))
    desk = build_incident_graph(settings, services, checkpointer=InMemorySaver())
    svc = _service(desk, settings)

    reply = await svc.report("d1", "search cpu saturation", "search", lead, control=control)

    assert reply.status == "drained"
    assert control.drain_requested and control.drain_reason == "SIGTERM"
    snapshot = await svc.snapshot("d1")
    assert snapshot.next == ("run_check", "run_check", "run_check")  # nothing lost
    assert services.remediator.applied == []  # type: ignore[attr-defined]

    reply = await svc.continue_run("d1", lead)  # a fresh worker picks it up

    assert reply.status == "completed"
    assert reply.output["status"] == "remediated"


async def test_time_travel_fork_with_a_corrected_severity(desk, settings, lead, services):  # type: ignore[no-untyped-def]
    svc = _service(desk, settings)
    reply = await svc.report("tt", "checkout returning 500 errors for customers", "checkout", lead)
    assert reply.status == "awaiting_input"  # sev2 needs approval

    history = await svc.history("tt")
    assert all(isinstance(s, StateSnapshot) for s in history)
    after_classify = next(s for s in history if s.next == ("fetch_metrics",))

    # "It was really a sev3": rewrite history right after `classify` ran.
    fork = await svc.fork("tt", after_classify, {"severity": "sev3"}, as_node="classify")
    reply = await svc.resume_from(fork, lead)

    assert reply.status == "completed"  # sev3 is auto-approved by policy
    assert reply.output["status"] == "remediated"
    assert services.remediator.applied == [("checkout", "failover_db")]  # type: ignore[attr-defined]
    # The original branch still exists in history: forks never overwrite.
    assert any(s.next == ("approve",) for s in await svc.history("tt"))


async def test_replaying_a_checkpoint_reruns_from_there(desk, settings, lead):  # type: ignore[no-untyped-def]
    svc = _service(desk, settings)
    await svc.report("rp", "search cpu saturation", "search", lead)
    before_assess = next(s for s in await svc.history("rp") if s.next == ("assess",))

    reply = await svc.resume_from(before_assess.config, lead)

    assert reply.output["status"] == "remediated"


async def test_bulk_import_from_another_tool(desk, settings, lead):  # type: ignore[no-untyped-def]
    svc = _service(desk, settings)

    await svc.import_incident(
        "imp",
        [
            [StateUpdate({"report": "legacy ticket: disk full", "service": "search"}, "intake")],
            [StateUpdate({"severity": "sev3", "category": "capacity"}, "classify")],
        ],
    )

    snapshot = await svc.snapshot("imp")
    assert snapshot.values["severity"] == "sev3"
    assert snapshot.next == ("fetch_metrics",)  # continues as if classify had run
    reply = await svc.continue_run("imp", lead)
    assert reply.output["status"] == "remediated"


async def test_clear_conversation_removes_all_messages(desk, settings, lead):  # type: ignore[no-untyped-def]
    svc = _service(desk, settings)
    await svc.report("gdpr", "search cpu saturation", "search", lead)
    assert (await svc.snapshot("gdpr")).values["messages"]

    await svc.clear_conversation("gdpr")

    assert (await svc.snapshot("gdpr")).values["messages"] == []
