"""Deprecated APIs still work, and each emits a specific deprecation class."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import task
from langgraph.graph import START, StateGraph
from langgraph.types import GraphOutput
from langgraph.warnings import (
    LangGraphDeprecatedSinceV05,
    LangGraphDeprecatedSinceV10,
    LangGraphDeprecatedSinceV11,
    LangGraphDeprecationWarning,
)

from lg_prod.graphs.runbook_agent import build_tools
from lg_prod.legacy import (
    deprecations_as_errors,
    legacy_echo_graph,
    legacy_interrupting_node,
    legacy_react_agent,
    legacy_ticket_validator,
    modern_echo_graph,
)
from tests.conftest import call, script, thread


def test_message_graph_matches_its_replacement():
    with pytest.warns(LangGraphDeprecatedSinceV10):
        legacy = legacy_echo_graph()

    old = legacy.invoke([HumanMessage("hi")])
    new = modern_echo_graph().invoke({"messages": [HumanMessage("hi")]})

    assert old[-1].content == new["messages"][-1].content == "ack: hi"


def test_create_react_agent_still_runs(services, lead):  # type: ignore[no-untyped-def]
    model = script(call("run_diagnostic", {"check": "db"}), "db failing")
    with pytest.warns(LangGraphDeprecatedSinceV10):
        agent = legacy_react_agent(model, build_tools(services)[:1])

    out = agent.invoke(
        {"messages": [HumanMessage("check db")], "service": "checkout"}, context=lead
    )
    assert out["messages"][-1].content == "db failing"


def test_validation_node_reports_schema_errors():
    with pytest.warns(LangGraphDeprecatedSinceV10):
        validator = legacy_ticket_validator()

    ok = call("IncidentTicket", {"service": "auth", "severity": 1}, "ok")
    bad = call("IncidentTicket", {"service": "auth", "severity": "high"}, "bad")
    out = validator.invoke({"messages": [AIMessage("", tool_calls=ok.tool_calls + bad.tool_calls)]})

    results = {m.tool_call_id: m for m in out["messages"] if isinstance(m, ToolMessage)}
    assert results["ok"].additional_kwargs.get("is_error") is not True
    assert results["bad"].additional_kwargs.get("is_error") is True


def test_node_interrupt_still_pauses_the_graph():
    builder = StateGraph(dict)
    builder.add_node("ask", legacy_interrupting_node)
    builder.add_edge(START, "ask")
    graph = builder.compile(checkpointer=InMemorySaver())

    with pytest.warns(LangGraphDeprecatedSinceV10):
        out = graph.invoke({"x": 1}, thread("ni"), version="v2")

    assert out.interrupts[0].value == {"question": "approve?"}


def test_v05_keyword_renames():
    with pytest.warns(LangGraphDeprecatedSinceV05):
        StateGraph(dict, input=dict)  # type: ignore[call-overload]
    with pytest.warns(LangGraphDeprecatedSinceV05):
        task(retry=None)(lambda: 1)  # type: ignore[call-overload]


def test_graph_output_dict_access_is_deprecated():
    builder = StateGraph(dict)
    builder.add_node("n", lambda s: {"x": 1})
    builder.add_edge(START, "n")
    out = builder.compile().invoke({"x": 0}, version="v2")
    assert isinstance(out, GraphOutput)

    with pytest.warns(LangGraphDeprecatedSinceV11):
        assert out["x"] == 1


def test_all_deprecations_share_one_base_class():
    for cls in (
        LangGraphDeprecatedSinceV05,
        LangGraphDeprecatedSinceV10,
        LangGraphDeprecatedSinceV11,
    ):
        assert issubclass(cls, LangGraphDeprecationWarning)
    with deprecations_as_errors(), pytest.raises(LangGraphDeprecationWarning):
        legacy_echo_graph()
