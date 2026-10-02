"""Hand-built tool-calling agent: ToolNode, tools_condition, injection, errors."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import tools_condition
from langgraph.store.memory import InMemoryStore

from lg_prod.graphs.runbook_agent import build_runbook_agent
from tests.conftest import call, script, thread


def _ask(service: str, text: str) -> dict:  # type: ignore[type-arg]
    return {"messages": [HumanMessage(text)], "service": service, "notes": []}


def test_tools_condition_routes_on_tool_calls():
    assert tools_condition([AIMessage("done")]) == "__end__"
    assert tools_condition({"messages": [call("add_note", {"note": "x"})]}) == "tools"


def test_injected_state_and_command_from_tool(services, lead):  # type: ignore[no-untyped-def]
    model = script(
        call("run_diagnostic", {"check": "db"}),
        call("add_note", {"note": "db is failing"}, "call_2"),
        "Failover recommended.",
    )
    log: list[str] = []
    agent = build_runbook_agent(model, services, checkpointer=InMemorySaver(), audit_log=log)

    out = agent.invoke(_ask("checkout", "why is checkout erroring?"), thread("a1"), context=lead)

    diag = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert diag.content == "db: FAILING (db failing)"  # service came from state
    assert out["notes"] == ["db is failing"]  # Command(update=...) from a tool
    assert out["messages"][-1].content == "Failover recommended."
    assert log == ["run_diagnostic:success", "add_note:command"]
    # The model never saw the injected arguments in the tool schemas.
    assert "run_diagnostic" in model.bound_tools[0]


def test_injected_store_reads_long_term_memory(services, lead):  # type: ignore[no-untyped-def]
    store = InMemoryStore()
    store.put(
        ("teams", "platform", "incidents"),
        "INC-1",
        {"service": "checkout", "summary": "checkout: db failover fixed it"},
    )
    model = script(call("recall_incidents", {"query": "db"}), "Seen before.")
    agent = build_runbook_agent(model, services, store=store)

    out = agent.invoke(_ask("checkout", "seen this?"), context=lead)

    assert out["messages"][2].content == "checkout: db failover fixed it"


def test_handled_tool_errors_are_shown_to_the_model(services, engineer):  # type: ignore[no-untyped-def]
    model = script(
        call("restart_service", {"reason": "db"}),
        call("page_human", {"message": " "}, "call_2"),
        "I could not restart it.",
    )
    agent = build_runbook_agent(model, services)

    out = agent.invoke(_ask("checkout", "restart it"), context=engineer)

    errors = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.status for m in errors] == ["error", "error"]
    assert "may not restart" in errors[0].content
    assert "must not be empty" in errors[1].content
    assert services.remediator.applied == []  # type: ignore[attr-defined]


def test_tool_runtime_gives_context_and_streams(services, lead):  # type: ignore[no-untyped-def]
    model = script(call("restart_service", {"reason": "oom"}), "Restarted.")
    agent = build_runbook_agent(model, services)

    custom = [
        chunk
        for chunk in agent.stream(_ask("search", "restart"), context=lead, stream_mode="custom")
    ]

    assert custom == [{"event": "restart", "service": "search"}]
    assert services.remediator.applied == [("search", "restart (oom)")]  # type: ignore[attr-defined]


def test_remaining_steps_stops_the_loop_gracefully(services, lead):  # type: ignore[no-untyped-def]
    endless = script(*[call("add_note", {"note": f"n{i}"}, f"c{i}") for i in range(50)])
    agent = build_runbook_agent(endless, services)

    out = agent.invoke(_ask("search", "loop"), {"recursion_limit": 8}, context=lead)

    assert out["messages"][-1].content.startswith("Step budget exhausted")
