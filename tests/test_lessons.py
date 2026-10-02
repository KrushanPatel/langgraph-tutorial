"""Production lessons verified against the installed LangGraph source (1.2.x).

Each test pins down one behaviour that is easy to get wrong. If an upgrade
changes one of them, the failing test points at the lesson to revisit.
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.callbacks import (
    GraphInterruptEvent,
    GraphResumeEvent,
    get_async_graph_callback_manager_for_config,
    get_sync_graph_callback_manager_for_config,
)
from langgraph.channels import UntrackedValue
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import TASKS
from langgraph.errors import (
    EmptyInputError,
    ErrorCode,
    GraphBubbleUp,
    GraphInterrupt,
    InvalidUpdateError,
    NodeCancelledError,
    NodeError,
    ParentCommand,
)
from langgraph.graph import START, StateGraph
from langgraph.runtime import BaseUser, ExecutionInfo, Runtime, ServerInfo
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Interrupt, Send, StreamWriter, interrupt

from lg_prod.context import DeskContext
from lg_prod.streaming import AuditCallbackHandler
from tests.conftest import thread


class S(TypedDict, total=False):
    findings: Annotated[list[str], operator.add]
    owner: str
    answers: Annotated[list[Any], operator.add]
    handle: Annotated[str, UntrackedValue(str)]


def _ok(state: S) -> S:
    return {"findings": ["ok"]}


def _boom(state: S) -> S:
    raise ValueError("boom")


def _handler(state: S, error: NodeError) -> S:
    return {"findings": [f"handled {error.node}: {error.error}"]}


# --------------------------------------------------------------- error handling


def test_error_handler_rescues_a_lone_failing_node():
    builder = StateGraph(S)
    builder.add_node("boom", _boom, error_handler=_handler)
    builder.add_edge(START, "boom")

    assert builder.compile().invoke({"findings": []})["findings"] == ["handled boom: boom"]


def test_error_handler_does_not_rescue_a_parallel_superstep():
    builder = StateGraph(S)
    builder.add_node("ok", _ok)
    builder.add_node("boom", _boom, error_handler=_handler)
    builder.add_edge(START, "ok")
    builder.add_edge(START, "boom")

    # Same handler, but a sibling ran in the same superstep: the run fails.
    # In fan-outs, catch inside the node (see `run_check` in triage.py).
    with pytest.raises(ValueError, match="boom"):
        builder.compile().invoke({"findings": []})


def test_parallel_writes_to_a_last_value_key_are_rejected():
    builder = StateGraph(S)
    builder.add_node("a", lambda s: {"owner": "a"})
    builder.add_node("b", lambda s: {"owner": "b"})
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")

    with pytest.raises(InvalidUpdateError) as info:
        builder.compile().invoke({})
    assert ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE.value in str(info.value)


def test_user_raised_cancellation_becomes_a_node_failure():
    async def cancels(state: S) -> S:
        raise asyncio.CancelledError()

    builder = StateGraph(S)
    builder.add_node("cancels", cancels)
    builder.add_edge(START, "cancels")

    with pytest.raises(NodeCancelledError):
        asyncio.run(builder.compile().ainvoke({}))


def test_empty_input_on_a_fresh_thread():
    builder = StateGraph(S)
    builder.add_node("ok", _ok)
    builder.add_edge(START, "ok")
    graph = builder.compile(checkpointer=InMemorySaver())

    # `None` means "continue this thread" - there is nothing to continue.
    with pytest.raises(EmptyInputError):
        graph.invoke(None, thread("new"))


# ------------------------------------------------------------------ interrupts


def test_broad_except_swallows_interrupts():
    def careless(state: S) -> S:
        try:
            answer = interrupt("approve?")
        except Exception:  # GraphInterrupt IS an Exception
            answer = "swallowed"
        return {"answers": [answer]}

    def careful(state: S) -> S:
        try:
            answer = interrupt("approve?")
        except GraphBubbleUp:  # interrupts, drains, parent commands
            raise
        except Exception:
            answer = "error"
        return {"answers": [answer]}

    for node, paused in ((careless, False), (careful, True)):
        builder = StateGraph(S)
        builder.add_node("n", node)
        builder.add_edge(START, "n")
        out = builder.compile(checkpointer=InMemorySaver()).invoke(
            {"answers": []}, thread(node.__name__), version="v2"
        )
        assert bool(out.interrupts) is paused
    assert issubclass(GraphInterrupt, GraphBubbleUp)


def test_state_written_before_an_interrupt_is_discarded():
    calls: list[str] = []

    def ask(state: S, writer: StreamWriter) -> S:
        calls.append("ran")
        writer({"progress": "asking"})  # streamed immediately, every time
        answer = interrupt("approve?")
        return {"answers": [answer]}

    builder = StateGraph(S)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    graph = builder.compile(checkpointer=InMemorySaver())
    cfg = thread("re")

    custom = list(graph.stream({"answers": []}, cfg, stream_mode="custom"))
    graph.invoke(Command(resume="yes"), cfg)

    assert custom == [{"progress": "asking"}]
    assert calls == ["ran", "ran"]  # the node re-ran from the top on resume
    assert graph.get_state(cfg).values["answers"] == ["yes"]


def test_resume_with_a_map_answers_parallel_interrupts():
    def ask(state: dict[str, Any]) -> S:
        return {"answers": [f"{state['who']}={interrupt(state['who'])}"]}

    builder = StateGraph(S)
    builder.add_node("ask", ask)
    builder.add_conditional_edges(START, lambda s: [Send("ask", {"who": w}) for w in ("db", "net")])
    graph = builder.compile(checkpointer=InMemorySaver())
    cfg = thread("map")

    pending: tuple[Interrupt, ...] = graph.invoke({"answers": []}, cfg, version="v2").interrupts
    assert TASKS in graph.channels  # `Send` packets travel on this channel
    graph.invoke(Command(resume={i.id: i.value.upper() for i in pending}), cfg)

    assert sorted(graph.get_state(cfg).values["answers"]) == ["db=DB", "net=NET"]


def test_empty_dict_resume_answers_nothing():
    builder = StateGraph(S)
    builder.add_node("ask", lambda s: {"answers": [interrupt("q")]})
    builder.add_edge(START, "ask")
    graph = builder.compile(checkpointer=InMemorySaver())
    cfg = thread("empty")
    graph.invoke({"answers": []}, cfg)

    # A dict resume is read as {interrupt_id: value}; {} answers no interrupt.
    out = graph.invoke(Command(resume={}), cfg, version="v2")
    assert out.interrupts
    out = graph.invoke(Command(resume={"ok": True}), cfg, version="v2")
    assert out.value["answers"] == [{"ok": True}]


def test_untracked_values_are_lost_on_resume():
    seen: list[str | None] = []

    def first(state: S) -> S:
        return {"handle": "db-connection-7"}

    def second(state: S) -> S:
        seen.append(state.get("handle"))
        interrupt("pause")
        return {}

    builder = StateGraph(S)
    builder.add_sequence([("first", first), ("second", second)])
    builder.add_edge(START, "first")
    graph = builder.compile(checkpointer=InMemorySaver())
    graph.invoke({}, thread("u"))
    graph.invoke(Command(resume=True), thread("u"))

    assert seen == ["db-connection-7", None]  # rebuilt from the checkpoint


def test_command_parent_at_the_root_escapes_to_the_caller():
    builder = StateGraph(S)
    builder.add_node("n", lambda s: Command(graph=Command.PARENT, goto="elsewhere"))
    builder.add_edge(START, "n")

    with pytest.raises(ParentCommand):
        builder.compile().invoke({})


# --------------------------------------------------------- runtime and callbacks


class ApiUser:
    """Implements the `BaseUser` protocol, as LangGraph Server auth would."""

    identity = "u-lead"
    display_name = "On-call lead"
    is_authenticated = True
    permissions: tuple[str, ...] = ("incidents:approve",)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __iter__(self) -> Any:
        return iter(("identity", "display_name", "is_authenticated", "permissions"))


def test_unit_testing_a_node_with_a_hand_built_runtime():
    """Nodes are plain functions: test them without compiling a graph."""

    def whoami(state: S, runtime: Runtime[DeskContext]) -> S:
        info = runtime.execution_info
        user = runtime.server_info.user if runtime.server_info else None
        return {
            "owner": f"{runtime.context.user_id} via {user.identity if user else 'local'}"
            f" attempt {info.node_attempt if info else 1}"
        }

    user: BaseUser = ApiUser()  # type: ignore[assignment]
    runtime = Runtime(
        context=DeskContext("u-lead", role="oncall_lead"),
        store=InMemoryStore(),
        execution_info=ExecutionInfo(
            checkpoint_id="c1", checkpoint_ns="", task_id="t1", node_attempt=2
        ),
        server_info=ServerInfo(assistant_id="a1", graph_id="incident_desk", user=user),
    )

    assert whoami({}, runtime) == {"owner": "u-lead via u-lead attempt 2"}
    retry = runtime.override(execution_info=ExecutionInfo("c1", "", "t1", node_attempt=3))
    assert whoami({}, retry)["owner"].endswith("attempt 3")


async def test_dispatching_graph_callbacks_directly():
    """The helpers LangGraph uses internally also make handlers unit-testable."""
    audit = AuditCallbackHandler()
    config = {"callbacks": [audit]}
    interrupt_event = GraphInterruptEvent(
        run_id=None,
        status="interrupt_before",
        checkpoint_id="c1",
        checkpoint_ns=(),
        interrupts=(Interrupt(value="approve?"),),
    )

    get_sync_graph_callback_manager_for_config(config).on_interrupt(interrupt_event)  # type: ignore[arg-type]
    await get_async_graph_callback_manager_for_config(config).on_resume(  # type: ignore[arg-type]
        GraphResumeEvent(run_id=None, status="input", checkpoint_id="c1", checkpoint_ns=())
    )

    assert audit.statuses() == [("interrupt", "interrupt_before", ()), ("resume", "input", ())]
