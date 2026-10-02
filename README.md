# LangGraph in production: a learning codebase

Production-style reference implementations of the **LangGraph 1.x** library. It answers one question: *how is each part of LangGraph actually used in production-ready code?*

This is the LangGraph companion to [langchain-tutorial](https://github.com/KrushanPatel/langchain-tutorial). That repo covers `langchain` (agents, middleware, tools); this one goes one layer down, to the graph runtime those agents run on: `StateGraph`, channels, Pregel, the Functional API, persistence, interrupts, streaming and deployment.

Every feature is used inside a realistic **on-call incident desk**, alongside the concerns that tutorials usually skip: authorization, retries and timeouts, graceful shutdown, approvals, time travel, encryption at rest, long-term memory and testing. Module docstrings explain **why** each decision is made, and the tests show each behaviour working.

The only API reference used is [reference.langchain.com/python/langgraph](https://reference.langchain.com/python/langgraph). Every API was checked against the installed source (`langgraph` 1.2.12, `langgraph-checkpoint` 4.2, `langgraph-prebuilt` 1.1).

## What is in it

| Graph | Built with | Highlights |
|---|---|---|
| **Incident desk** (`graphs/triage.py`) | `StateGraph` | Input/output/private schemas, `Send` map-reduce, `Command` routing, deferred nodes, retry/timeout/cache policies, error handlers, human approval via `interrupt()`, generative-UI messages, long-term memory |
| **Escalation** (`graphs/escalation.py`) | subgraph | Its own state schema, an interrupt inside a subgraph, `Command.PARENT` hand-back |
| **Runbook agent** (`graphs/runbook_agent.py`) | `ToolNode` + `tools_condition` | A hand-built tool-calling loop: `InjectedState`, `InjectedStore`, `ToolRuntime`, tools returning `Command`, tool-error policy, `wrap_tool_call` |
| **Post-mortem writer** (`graphs/postmortem.py`) | Functional API | `@entrypoint`/`@task`, parallel futures, `previous` + `entrypoint.final`, heartbeat idle timeouts |
| **Alert pipeline** (`graphs/alert_pipeline.py`) | raw `Pregel` + `NodeBuilder` | Every channel type, plus a custom `BaseChannel` |

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q                 # full suite, offline: no API key needed
uv run python -m lg_prod         # interactive incident desk (offline demo backends)
uv run python -m lg_prod --llm   # same, with an LLM classifier (needs ANTHROPIC_API_KEY)
uv run python scripts/api_coverage.py --missing
```

All settings live in `src/lg_prod/config.py`. Override any of them with `LG_`-prefixed environment variables, for example `LG_PERSISTENCE=sqlite` or `LG_DURABILITY=sync`. Copy `.env.example` to `.env` to get started.

## Where each LangGraph module lives

Read roughly in this order:

| LangGraph module | File(s) | What it shows |
|---|---|---|
| — | `config.py`, `context.py`, `services.py` | Typed settings; run **context** (identity, never checkpointed) vs. **state**; backends behind Protocols |
| `langgraph.graph`, `.channels`, `.managed`, `.graph.message` | `state.py` | Reducers (`operator.add`, `add_messages`, a custom upsert), `EphemeralValue`, `UntrackedValue`, `DeltaChannel`, `RemainingSteps`/`IsLastStep`, input/output/private schemas |
| `langgraph.graph` (`StateGraph`) | `graphs/triage.py` | Nodes with `input_schema`, `defer`, `metadata`, `destinations`; `set_node_defaults`; `add_sequence`; conditional edges; static breakpoints; `get_graph(xray=True)` |
| `langgraph.types` | `graphs/triage.py`, `service.py` | `Send`, `Command` (`goto`/`update`/`resume`/`PARENT`), `Overwrite`, `interrupt(response_schema=...)`, `RetryPolicy`, `CachePolicy`, `TimeoutPolicy`, `TracePolicy`/`omit_payload`, `GraphOutput`, `StateSnapshot`, `StateUpdate`, `PregelTask` |
| `langgraph.errors` | `graphs/triage.py`, `tests/test_lessons.py` | Node-level `error_handler` + `NodeError`, `NodeTimeoutError`, `GraphRecursionError`, `GraphDrained`, `ParentCommand`, `GraphBubbleUp` handling |
| `langgraph.runtime`, `langgraph.config` | `graphs/*.py`, `tests/test_lessons.py` | `Runtime` (context, store, stream writer, `execution_info.node_attempt`, heartbeat), `get_runtime`/`get_config`/`get_store`/`get_stream_writer`, `RunControl`, unit-testing nodes with a hand-built `Runtime` |
| `langgraph.graph.ui` | `graphs/triage.py` | `push_ui_message` / `delete_ui_message` merged by `ui_message_reducer` |
| `langgraph.prebuilt` | `graphs/runbook_agent.py` | `ToolNode`, `tools_condition`, `InjectedState`, `InjectedStore`, `ToolRuntime.emit_output_delta` |
| `langgraph.func` | `graphs/postmortem.py` | `@entrypoint` (checkpointer/store/cache/context), `@task` with policies, futures, `entrypoint.final` |
| `langgraph.pregel`, `.channels` | `graphs/alert_pipeline.py` | `Pregel`, `NodeBuilder` (`subscribe_to`/`read_from`/`write_to`/`meta`), all 10 built-in channels, a custom channel, `TAG_HIDDEN` |
| `langgraph.checkpoint.*`, `.store.*`, `.cache.*` | `persistence.py` | `InMemorySaver`/`AsyncSqliteSaver`, `JsonPlusSerializer` allowlist, `EncryptedSerializer` (AES), `InMemoryStore`/`SqliteStore` with `IndexConfig` semantic search and `TTLConfig`, `InMemoryCache`/`SqliteCache` |
| `langgraph.stream`, `langgraph.callbacks` | `streaming.py` | Typed v2 `StreamPart`s for all 7 modes, `stream_events(version="v3")` run streams and transformers (custom + built-in), `GraphCallbackHandler` |
| `langgraph.pregel.remote` | `remote.py` | `RemoteGraph` as a client and as a node of a local graph |
| serving | `service.py` | Thread ids, interrupts as API states, answering several interrupts by id, drain/continue, time travel (fork/replay), bulk import, erasing messages |
| `langgraph.warnings` | `legacy.py` | Deprecated APIs (`MessageGraph`, `create_react_agent`, `ValidationNode`, `NodeInterrupt`, ...) next to their replacements |

## Production lessons found in the source

Each of these was verified against the installed LangGraph code and has a test (most are in `tests/test_lessons.py`):

**Execution and errors**
- **Node `error_handler`s only rescue a failure when the failing task runs alone in its superstep.** In a `Send` fan-out or a parallel branch, the run still fails. Catch inside fan-out nodes, and use `ExecutionInfo.node_attempt` to degrade gracefully on the last retry.
- **An error handler is a separate node.** Returning a plain dict does *not* follow the failed node's outgoing edges, so route with `Command(goto=...)`.
- **Node timeouts are async-only.** A sync node with `timeout=` is rejected at compile time. Calling sync `invoke` on a graph containing an async node with a timeout makes that node raise `ValueError` on every run, which an error handler can silently swallow.
- **Two parallel writes to a plain (`LastValue`) key raise `InvalidUpdateError`.** Give the key a reducer.
- **A node that raises `asyncio.CancelledError` itself becomes `NodeCancelledError`**, not a silent cancellation.
- **`RemainingSteps` lets a loop stop gracefully** before `GraphRecursionError`.

**Interrupts and human-in-the-loop**
- **`except Exception` around `interrupt()` swallows the pause**: `GraphInterrupt` is an `Exception`. Re-raise `GraphBubbleUp` first.
- **A node re-runs from the top on resume, and its state writes before `interrupt()` are discarded** (stream-writer events are still emitted, again on each run). Push UI cards in a node that completes.
- **A dict resume value is read as `{interrupt_id: value}`.** `Command(resume={})` answers nothing and the graph interrupts again. The same mechanism answers several parallel interrupts at once.
- **`interrupt(response_schema=...)` validates the resume value** (pydantic `ValidationError`) and sends the JSON schema to the client.
- **`Command(graph=Command.PARENT)` at the root escapes to the caller as `ParentCommand`.** Inside a subgraph, don't annotate parent targets in `Command[Literal[...]]`: return-type literals declare edges of the subgraph itself, and compilation fails.

**State and channels**
- **Private keys reach a node only through its `input_schema`.** Nodes reading a private key must declare a schema that contains it.
- **`UntrackedValue` is never checkpointed, so it is gone after a resume.** `AnyValue` clears itself in any step without writes.
- **`get_graph()` dry-runs task preparation with an empty state**, so a `CachePolicy.key_func` must tolerate missing keys.
- **`CachePolicy` does nothing without `compile(cache=...)`**, and no warning is raised.
- **A custom channel's `from_checkpoint` receives a private `MISSING` sentinel** on a fresh thread, not `None`.

**Persistence**
- **`SqliteSaver` is sync-only**; graphs run with `ainvoke` need `AsyncSqliteSaver`. Don't point the checkpointer and the store at one SQLite file: two concurrent writers fail with "database is locked".
- **Checkpoint serialization is moving to an allowlist.** Unregistered custom types log a warning today and will be blocked in a future version, so list them in `allowed_msgpack_modules`.
- **`InMemoryStore` raises on `ttl=`.** Check `store.supports_ttl`. In `SqliteStore`, expired items stay readable until `sweep_ttl()` runs.
- **Ops in one `store.batch()` are not applied in order**: gets and searches don't see that batch's puts.
- **Under the Functional API, `get_state().values` is the last *returned* value**, while `previous` is what `entrypoint.final(save=...)` stored.

**Streaming**
- **`stream_mode="messages"` also streams messages that nodes write to state**, not just LLM tokens. Tag internal LLM calls with `TAG_NOSTREAM`.
- **v3 run-stream projections are lazy single-consumer channels**: subscribe to everything you need *before* driving the run, or the items are dropped.
- **`LifecycleTransformer`/`SubgraphTransformer` are always registered**; passing them again raises a conflict. Under v3, message `content` comes back as content blocks.

## Testing

Tests run the real graphs, channels, checkpointers and stores; only LLMs are replaced by `ScriptedChatModel` (`tests/conftest.py`). `RemoteGraph` is tested against a loopback client that speaks the SDK's `runs.stream` protocol.

```bash
uv run pytest -q                                  # everything offline (~7s)
uv run pytest tests/test_lessons.py -v            # the gotchas above
uv run ruff check src tests scripts && uv run ruff format src tests scripts
uv run mypy src                                   # strict mode
```

## Coverage and limitations

- **169 of 171 (98.8%) public LangGraph symbols** are imported and used by the code or tests, measured by `scripts/api_coverage.py` (an AST scan, so comments don't count) and enforced by `tests/test_api_coverage.py` (≥ 90%). The denominator is the `__all__` of every public `langgraph.*` module, plus the public classes of `checkpoint.*`, `store.*` and `cache.*`. The two unused symbols, `PregelExecutableTask` and `TaskNotFound`, belong to the distributed executor.
- Excluded from the denominator: `langgraph._internal`, private `langgraph.pregel._*` modules and `langgraph.cache.redis`. By options and behaviours, coverage is lower than by symbol.
- `DeltaChannel`, `stream_events(version="v3")` and node-level error handlers are beta/experimental upstream and may change. Several lessons above pin down 1.2.x behaviour on purpose.
- Persistence uses in-memory and SQLite backends. In production, use Postgres (`langgraph-checkpoint-postgres`); the interfaces are identical.
- The LLM paths are tested with a scripted fake. Provider-specific behaviour has not been verified against a real model.
- macOS note: if `import lg_prod` fails outside pytest, the editable-install `.pth` file may carry the `hidden` flag, which Python 3.13 skips. Run `chflags nohidden .venv/lib/python3.13/site-packages/*.pth`.
