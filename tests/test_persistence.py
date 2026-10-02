"""Checkpointers, serialization, encryption, stores (search/TTL/batch ops), caches."""

from __future__ import annotations

import logging
import operator
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.cache.sqlite import SqliteCache
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    copy_checkpoint,
    create_checkpoint,
    empty_checkpoint,
    get_checkpoint_id,
)
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.config import get_config
from langgraph.constants import CONF, CONFIG_KEY_CHECKPOINTER
from langgraph.func import entrypoint, task
from langgraph.graph import START, StateGraph
from langgraph.store.base import (
    GetOp,
    InvalidNamespaceError,
    Item,
    ListNamespacesOp,
    MatchCondition,
    NamespaceMatchType,
    NamespacePath,
    Op,
    PutOp,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import AsyncSqliteStore, SqliteStore
from langgraph.types import CachePolicy, Command, Durability

from lg_prod.config import Settings
from lg_prod.graphs.triage import build_incident_graph
from lg_prod.models import Classification
from lg_prod.persistence import (
    HashingEmbeddings,
    build_serializer,
    incident_index,
    open_persistence,
    validate_checkpointer,
)
from lg_prod.services import demo_services
from tests.conftest import thread

AES_KEY = "0123456789abcdef0123456789abcdef"
NS = ("teams", "platform", "incidents")
SEARCH_CPU = {"report": "search cpu saturation", "service": "search"}


@pytest.fixture
def sqlite_settings(tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        persistence="sqlite",
        sqlite_path=str(tmp_path / "desk.sqlite"),
        LANGGRAPH_AES_KEY=AES_KEY,
    )


# ------------------------------------------------------------ checkpointers


async def test_encrypted_sqlite_checkpoints_survive_a_restart(sqlite_settings, lead):  # type: ignore[no-untyped-def]
    report = {"report": "checkout returning 500 errors for customers", "service": "checkout"}
    async with open_persistence(sqlite_settings) as p:
        desk = build_incident_graph(sqlite_settings, demo_services(), checkpointer=p.checkpointer)
        await desk.ainvoke(report, thread("db1"), context=lead)  # pauses for approval

    with closing(sqlite3.connect(sqlite_settings.sqlite_path)) as conn:
        raw = conn.execute("select type, checkpoint from checkpoints").fetchall()
    assert raw and all(row[0].endswith("+aes") for row in raw)
    assert all(b"checkout" not in row[1] for row in raw)  # encrypted at rest

    # "Restart": brand-new connection and graph object, same thread id.
    async with open_persistence(sqlite_settings) as p:
        desk = build_incident_graph(
            sqlite_settings, demo_services(), checkpointer=p.checkpointer, store=p.store
        )
        assert (await desk.aget_state(thread("db1"))).next == ("approve",)
        out = await desk.ainvoke(
            Command(resume={"approved": True, "note": "ok"}), thread("db1"), context=lead
        )
        assert out["status"] == "remediated"
        assert p.store.search(lead.memory_namespace, query="checkout db")[0].key == "INC-db1"


async def test_sync_sqlite_saver_cannot_serve_async_runs(tmp_path: Path, settings, lead):  # type: ignore[no-untyped-def]
    with SqliteSaver.from_conn_string(str(tmp_path / "sync.sqlite")) as saver:
        desk = build_incident_graph(settings, demo_services(), checkpointer=saver)
        with pytest.raises(NotImplementedError, match="does not support async"):
            await desk.ainvoke(SEARCH_CPU, thread("x"), context=lead)


async def test_async_sqlite_checkpointer(sqlite_settings, lead):  # type: ignore[no-untyped-def]
    async with open_persistence(sqlite_settings) as p:
        assert isinstance(p.checkpointer, AsyncSqliteSaver)
        desk = build_incident_graph(sqlite_settings, demo_services(), checkpointer=p.checkpointer)
        out = await desk.ainvoke(SEARCH_CPU, thread("a1"), context=lead)
        history = [s async for s in desk.aget_state_history(thread("a1"))]
    assert out["status"] == "remediated"
    assert len(history) > 5


def test_serializer_allowlist(caplog):  # type: ignore[no-untyped-def]
    value = Classification(severity="sev1", category="errors", rationale="r")
    default = JsonPlusSerializer()
    with caplog.at_level(logging.WARNING):
        default.loads_typed(default.dumps_typed(value))
    assert "unregistered type lg_prod.models.Classification" in caplog.text

    caplog.clear()
    allowed = build_serializer(Settings(_env_file=None))  # type: ignore[call-arg]
    with caplog.at_level(logging.WARNING):
        assert allowed.loads_typed(allowed.dumps_typed(value)) == value
    assert caplog.text == ""


def test_checkpoint_primitives():
    saver = InMemorySaver()
    assert MemorySaver is InMemorySaver  # legacy alias
    base: Checkpoint = empty_checkpoint()
    snapshot = create_checkpoint(copy_checkpoint(base), None, step=1)
    assert snapshot["id"] != base["id"]

    meta: CheckpointMetadata = {"source": "input", "step": 1, "parents": {}}
    cfg = saver.put({"configurable": {"thread_id": "raw", "checkpoint_ns": ""}}, snapshot, meta, {})

    assert get_checkpoint_id(cfg) == snapshot["id"]
    loaded = saver.get_tuple(cfg)
    assert isinstance(loaded, CheckpointTuple)
    assert loaded.metadata["step"] == 1
    assert [t.config for t in saver.list({"configurable": {"thread_id": "raw"}})] == [cfg]


def test_checkpointer_validation():
    assert validate_checkpointer(None) is None
    with pytest.raises(TypeError, match="Invalid checkpointer"):
        validate_checkpointer(InMemorySaver)  # the class, not an instance


def test_nodes_can_reach_the_checkpointer_through_config():
    seen: list[Any] = []

    def node(state: dict[str, Any]) -> dict[str, Any]:
        seen.append(get_config()[CONF][CONFIG_KEY_CHECKPOINTER])
        return state

    saver = InMemorySaver()
    builder = StateGraph(dict)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.compile(checkpointer=saver).invoke({"x": 1}, thread("c"))
    assert seen == [saver]


@pytest.mark.parametrize(("durability", "max_checkpoints"), [("sync", 99), ("exit", 1)])
def test_durability_controls_checkpoint_writes(durability: Durability, max_checkpoints: int):
    class S(TypedDict):
        n: Annotated[list[int], operator.add]

    builder = StateGraph(S)
    builder.add_sequence([("a", lambda s: {"n": [1]}), ("b", lambda s: {"n": [2]})])
    builder.add_edge(START, "a")
    saver = InMemorySaver()
    graph = builder.compile(checkpointer=saver)

    graph.invoke({"n": []}, thread("d"), durability=durability)

    written = len(list(saver.list(thread("d"))))
    assert written <= max_checkpoints
    if durability == "sync":
        assert written >= 3  # input + after a + after b


# ------------------------------------------------------------------ stores


def test_semantic_search_filter_and_namespaces():
    store = InMemoryStore(index=incident_index())
    store.put(NS, "INC-1", {"service": "checkout", "summary": "checkout database failover"})
    store.put(NS, "INC-2", {"service": "search", "summary": "search cpu scale out"})
    store.put(("teams", "payments", "incidents"), "INC-3", {"service": "pay", "summary": "x"})
    store.put(NS, "INC-4", {"service": "search", "summary": "noise"}, index=False)

    hits: list[SearchItem] = store.search(("teams",), query="database failover on checkout")
    assert hits[0].key == "INC-1" and hits[0].score is not None
    assert [h.key for h in store.search(NS, filter={"service": "search"})] == ["INC-2", "INC-4"]

    prefix: NamespacePath = ("teams", "*", "incidents")
    assert len(store.list_namespaces(prefix=prefix)) == 2
    assert store.list_namespaces(prefix=("teams",), max_depth=2) == [
        ("teams", "payments"),
        ("teams", "platform"),
    ]
    item = store.get(NS, "INC-1")
    assert isinstance(item, Item) and item.value["service"] == "checkout"


def test_batch_operations():
    store = InMemoryStore()
    match_type: NamespaceMatchType = "prefix"
    ops: list[Op] = [
        PutOp(NS, "INC-1", {"service": "auth"}),
        GetOp(NS, "INC-1"),
        SearchOp(NS, filter={"service": "auth"}),
        ListNamespacesOp(match_conditions=(MatchCondition(match_type, ("teams",)),)),
    ]
    results = store.batch(ops)
    put_result, got, found, namespaces = results
    assert put_result is None
    # Ops in one batch are NOT applied in list order. InMemoryStore runs gets
    # and searches before the batch's puts, but lists namespaces after them.
    # Never rely on read-your-writes inside a single batch.
    assert got is None
    assert found == []
    assert namespaces == [NS]
    [after] = store.batch([GetOp(NS, "INC-1")])
    assert isinstance(after, Item) and after.key == "INC-1"


def test_namespace_validation():
    store = InMemoryStore()
    for bad in [(), ("a.b",), ("",), ("langgraph",)]:
        with pytest.raises(InvalidNamespaceError):
            store.put(bad, "k", {})


def test_index_path_helpers():
    doc = {"incident": {"title": "db down", "tags": ["db", "sev1"]}}
    assert tokenize_path("incident.tags[*]") == ["incident", "tags", "[*]"]
    assert get_text_at_path(doc, "incident.tags[*]") == ["db", "sev1"]
    assert get_text_at_path(doc, "incident.title") == ["db down"]

    embed = ensure_embeddings(lambda texts: [[float(len(t))] for t in texts])
    assert embed.embed_query("four") == [4.0]
    assert HashingEmbeddings(8).embed_query("a") != HashingEmbeddings(8).embed_query("b")


def test_in_memory_store_rejects_ttl():
    store = InMemoryStore()
    assert store.supports_ttl is False
    with pytest.raises(NotImplementedError, match="TTL is not supported"):
        store.put(NS, "k", {"v": 1}, ttl=5)


def test_sqlite_store_ttl_expires_items(tmp_path: Path):
    ttl = TTLConfig(default_ttl=0.001, refresh_on_read=False)  # minutes
    with SqliteStore.from_conn_string(str(tmp_path / "s.sqlite"), ttl=ttl) as store:
        store.setup()
        assert store.supports_ttl
        store.put(NS, "short", {"v": 1})
        store.put(NS, "long", {"v": 2}, ttl=60)
        time.sleep(1.1)  # expiry is swept at ~1s resolution
        assert store.get(NS, "short") is not None  # expired, but not swept yet
        assert store.sweep_ttl() == 1
        assert store.get(NS, "short") is None
        assert store.get(NS, "long") is not None


async def test_async_sqlite_store(tmp_path: Path):
    async with AsyncSqliteStore.from_conn_string(str(tmp_path / "a.sqlite")) as store:
        await store.setup()
        await store.aput(NS, "k", {"service": "auth"})
        assert (await store.aget(NS, "k")).value == {"service": "auth"}  # type: ignore[union-attr]


# ------------------------------------------------------------------ caches


def test_sqlite_cache_survives_a_new_graph_instance(tmp_path: Path):
    calls: list[int] = []

    def build() -> Any:
        @task(cache_policy=CachePolicy(ttl=60))
        def expensive(x: int) -> int:
            calls.append(x)
            return x * 10

        @entrypoint(cache=SqliteCache(path=str(tmp_path / "cache.sqlite")))
        def wf(x: int) -> int:
            return expensive(x).result()

        return wf

    assert build().invoke(4) == 40
    assert build().invoke(4) == 40  # new process, same cache file
    assert calls == [4]


def test_sqlite_saver_direct_usage(tmp_path: Path):
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        builder = StateGraph(dict)
        builder.add_node("n", lambda s: {"x": 1})
        builder.add_edge(START, "n")
        graph = builder.compile(checkpointer=saver)
        graph.invoke({"x": 0}, thread("s"))
        assert graph.get_state(thread("s")).values == {"x": 1}
