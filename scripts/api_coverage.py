"""Measure how much of LangGraph's public API this repository actually uses.

Denominator: `PUBLIC_API` below - the `__all__` of every public `langgraph.*`
module, plus hand-picked public names for modules that define no `__all__`
(`config`, `checkpoint.*`, `store.memory`, `cache.*`, the SQLite backends).

Excluded on purpose (listed so the number is honest):
* `langgraph._internal.*` and private `langgraph.pregel._*` modules.
* `langgraph.cache.redis` (needs a Redis server).
* Pure type aliases that cannot be "used" except in annotations are still
  counted - annotating with them is how they are used.

Numerator: a symbol counts only when a file under `src/` or `tests/` imports it
from a `langgraph` module *and* references it outside the import statement.
This is an AST scan, so names in comments or strings do not count.

Run: `uv run python scripts/api_coverage.py [--missing]`
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PUBLIC_API: dict[str, tuple[str, ...]] = {
    "langgraph.graph": (
        "END",
        "START",
        "StateGraph",
        "add_messages",
        "MessagesState",
        "MessageGraph",
    ),
    "langgraph.graph.message": ("REMOVE_ALL_MESSAGES", "push_message"),
    "langgraph.graph.state": ("CompiledStateGraph",),
    "langgraph.graph.ui": (
        "UIMessage",
        "RemoveUIMessage",
        "AnyUIMessage",
        "push_ui_message",
        "delete_ui_message",
        "ui_message_reducer",
    ),
    "langgraph.types": (
        "All",
        "Checkpointer",
        "StreamMode",
        "StreamWriter",
        "StreamPart",
        "ValuesStreamPart",
        "UpdatesStreamPart",
        "MessagesStreamPart",
        "CustomStreamPart",
        "CheckpointStreamPart",
        "TasksStreamPart",
        "DebugStreamPart",
        "TaskPayload",
        "TaskResultPayload",
        "CheckpointTask",
        "CheckpointPayload",
        "DebugPayload",
        "RetryPolicy",
        "TimeoutPolicy",
        "CachePolicy",
        "TracePolicy",
        "omit_payload",
        "Interrupt",
        "StateUpdate",
        "PregelTask",
        "PregelExecutableTask",
        "StateSnapshot",
        "Send",
        "Command",
        "Durability",
        "interrupt",
        "Overwrite",
        "GraphOutput",
        "ensure_valid_checkpointer",
    ),
    "langgraph.func": ("task", "entrypoint"),
    "langgraph.channels": (
        "BaseChannel",
        "AnyValue",
        "LastValue",
        "LastValueAfterFinish",
        "UntrackedValue",
        "EphemeralValue",
        "BinaryOperatorAggregate",
        "DeltaChannel",
        "NamedBarrierValue",
        "NamedBarrierValueAfterFinish",
        "Topic",
    ),
    "langgraph.config": ("get_config", "get_store", "get_stream_writer"),
    "langgraph.runtime": (
        "BaseUser",
        "ExecutionInfo",
        "RunControl",
        "Runtime",
        "ServerInfo",
        "get_runtime",
    ),
    "langgraph.errors": (
        "EmptyChannelError",
        "ErrorCode",
        "GraphDrained",
        "GraphRecursionError",
        "InvalidUpdateError",
        "GraphBubbleUp",
        "GraphInterrupt",
        "NodeCancelledError",
        "NodeError",
        "NodeInterrupt",
        "NodeTimeoutError",
        "ParentCommand",
        "EmptyInputError",
        "TaskNotFound",
        "create_error_message",
    ),
    "langgraph.managed": ("IsLastStep", "RemainingSteps"),
    "langgraph.constants": (
        "TAG_NOSTREAM",
        "TAG_HIDDEN",
        "CONF",
        "TASKS",
        "CONFIG_KEY_CHECKPOINTER",
    ),
    "langgraph.store.base": (
        "BaseStore",
        "Item",
        "Op",
        "PutOp",
        "GetOp",
        "SearchOp",
        "ListNamespacesOp",
        "MatchCondition",
        "NamespacePath",
        "NamespaceMatchType",
        "Embeddings",
        "ensure_embeddings",
        "tokenize_path",
        "get_text_at_path",
        "IndexConfig",
        "SearchItem",
        "TTLConfig",
        "InvalidNamespaceError",
    ),
    "langgraph.store.memory": ("InMemoryStore",),
    "langgraph.store.sqlite": ("SqliteStore", "AsyncSqliteStore"),
    "langgraph.checkpoint.base": (
        "BaseCheckpointSaver",
        "Checkpoint",
        "CheckpointMetadata",
        "CheckpointTuple",
        "SerializerProtocol",
        "empty_checkpoint",
        "copy_checkpoint",
        "create_checkpoint",
        "get_checkpoint_id",
    ),
    "langgraph.checkpoint.memory": ("InMemorySaver", "MemorySaver"),
    "langgraph.checkpoint.sqlite": ("SqliteSaver",),
    "langgraph.checkpoint.sqlite.aio": ("AsyncSqliteSaver",),
    "langgraph.checkpoint.serde.jsonplus": ("JsonPlusSerializer",),
    "langgraph.checkpoint.serde.encrypted": ("EncryptedSerializer",),
    "langgraph.cache.base": ("BaseCache",),
    "langgraph.cache.memory": ("InMemoryCache",),
    "langgraph.cache.sqlite": ("SqliteCache",),
    "langgraph.prebuilt": (
        "create_react_agent",
        "ToolNode",
        "ToolCallTransformer",
        "tools_condition",
        "ValidationNode",
        "InjectedState",
        "InjectedStore",
        "ToolRuntime",
    ),
    "langgraph.pregel": ("Pregel", "NodeBuilder"),
    "langgraph.pregel.remote": ("RemoteGraph", "RemoteException"),
    "langgraph.callbacks": (
        "GraphCallbackHandler",
        "GraphInterruptEvent",
        "GraphLifecycleEvent",
        "GraphLifecycleStatus",
        "GraphResumeEvent",
        "get_async_graph_callback_manager_for_config",
        "get_sync_graph_callback_manager_for_config",
    ),
    "langgraph.stream": (
        "AsyncGraphRunStream",
        "AsyncSubgraphRunStream",
        "CheckpointsTransformer",
        "CustomTransformer",
        "DebugTransformer",
        "GraphRunStream",
        "LifecyclePayload",
        "LifecycleTransformer",
        "ProtocolEvent",
        "StreamChannel",
        "StreamTransformer",
        "SubgraphRunStream",
        "SubgraphStatus",
        "SubgraphTransformer",
        "TasksTransformer",
        "UpdatesTransformer",
    ),
    "langgraph.warnings": (
        "LangGraphDeprecationWarning",
        "LangGraphDeprecatedSinceV05",
        "LangGraphDeprecatedSinceV10",
        "LangGraphDeprecatedSinceV11",
    ),
}


def all_symbols() -> set[str]:
    return {name for names in PUBLIC_API.values() for name in names}


def _used_in_file(path: Path) -> set[str]:
    """Public names imported from `langgraph*` and referenced outside the import."""
    tree = ast.parse(path.read_text(), filename=str(path))
    targets = all_symbols()
    aliases: dict[str, str] = {}  # local name -> public symbol
    module_aliases: set[str] = set()  # `import langgraph.types as lt`
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("langgraph"):
            for a in node.names:
                if a.name in targets:
                    aliases[a.asname or a.name] = a.name
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("langgraph"):
                    module_aliases.add(a.asname or a.name.split(".")[0])
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in aliases:
            used.add(aliases[node.id])
        elif isinstance(node, ast.Attribute) and node.attr in targets:
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in module_aliases:
                used.add(node.attr)
    # Names used only inside quoted annotations: parse them too.
    for node in ast.walk(tree):
        ann = getattr(node, "annotation", None) or getattr(node, "returns", None)
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            for sub in ast.walk(ast.parse(ann.value, mode="eval")):
                if isinstance(sub, ast.Name) and sub.id in aliases:
                    used.add(aliases[sub.id])
    return used


def used_symbols() -> set[str]:
    used: set[str] = set()
    for folder in ("src", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            used |= _used_in_file(path)
    return used


def report() -> tuple[int, int, dict[str, list[str]]]:
    used = used_symbols()
    missing = {
        mod: [n for n in names if n not in used]
        for mod, names in PUBLIC_API.items()
        if any(n not in used for n in names)
    }
    total = len(all_symbols())
    return len(used & all_symbols()), total, missing


def main() -> None:
    covered, total, missing = report()
    print(f"LangGraph public API coverage: {covered}/{total} = {covered / total:.1%}")
    if "--missing" in sys.argv:
        for mod, names in missing.items():
            print(f"  {mod}: {', '.join(names)}")


if __name__ == "__main__":
    main()
