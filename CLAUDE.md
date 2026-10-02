# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A learning project: production-style reference implementations of the LangGraph 1.x public API, built around an on-call **incident desk**. Companion to the `langchain-tutorial` repo, which covers `langchain` agents and middleware. The only API reference to follow is https://reference.langchain.com/python/langgraph. Verify APIs against the installed source in `.venv/lib/python3.13/site-packages/langgraph/`, not against older (0.x) examples. Module docstrings are the teaching material: keep them explaining *why* when changing code.

## Commands

```bash
uv sync
uv run pytest -q                                   # all tests, offline
uv run pytest tests/test_lessons.py::test_broad_except_swallows_interrupts
uv run ruff check src tests scripts && uv run ruff format src tests scripts
uv run mypy src                                    # strict mode
uv run python scripts/api_coverage.py --missing    # public-API coverage (must stay >= 90%)
uv run python -m lg_prod [--llm]                   # interactive demo
```

## Architecture (`src/lg_prod/`)

- `config.py`: `Settings` (pydantic-settings, prefix `LG_`). `context.py`: `DeskContext`, passed as `context=` and never checkpointed; authorization reads it. `state.py`: all schemas, reducers and special channels. `services.py`: backends behind Protocols, with in-memory fakes.
- `graphs/triage.py`: `build_incident_graph(settings, services, *, classifier, checkpointer, store, cache, interrupt_before/after)`, the main `StateGraph`. `graphs/escalation.py`: subgraph using `Command.PARENT`. `graphs/runbook_agent.py`: ToolNode loop. `graphs/postmortem.py`: Functional API. `graphs/alert_pipeline.py`: raw Pregel with every channel type.
- `persistence.py`: `open_persistence(settings)` (async CM) gives checkpointer/store/cache, memory or SQLite, with an encrypted, allowlisted serializer. `streaming.py`: v2 `StreamDigest`, v3 `CheckResultsTransformer`, `AuditCallbackHandler`. `service.py`: `IncidentDeskService` API boundary. `remote.py`: `RemoteGraph`. `legacy.py`: deprecated APIs next to their replacements.
- `scripts/api_coverage.py`: AST-based coverage of the public API (the denominator is hard-coded in `PUBLIC_API`).

## Gotchas (each has a test; see README "Production lessons")

- The desk has an async node with `timeout=`, so drive it with `ainvoke`/`astream`. Sync `invoke` silently degrades `fetch_metrics` through its error handler. `SqliteSaver` is sync-only; use `AsyncSqliteSaver`.
- Node `error_handler`s don't rescue failures in parallel supersteps. A handler is its own node, so route with `Command(goto=...)`.
- State writes before `interrupt()` are discarded, and the node re-runs on resume. Never wrap `interrupt()` in `except Exception` without re-raising `GraphBubbleUp`. A dict `resume` is a map `{interrupt_id: value}`, so `{}` resumes nothing.
- Private keys (`Scratchpad`) reach a node only through its `input_schema`. Cache `key_func`s must tolerate empty state, because `get_graph()` calls them.
- In subgraphs, don't put `Command.PARENT` targets in `Command[Literal[...]]` return annotations.
- v3 stream projections must be subscribed before the run is driven.

## Tests

`tests/conftest.py` provides `ScriptedChatModel` (streams tool calls), `script(...)`/`call(...)`, `thread(id)`, and `desk`/`lead`/`engineer`/`services`/`store` fixtures. Tests use real graphs and persistence; only LLMs are faked. `pythonpath = ["src", "."]` is set in pytest config.
