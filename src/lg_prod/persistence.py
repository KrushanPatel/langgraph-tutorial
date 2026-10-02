"""Persistence: checkpointers (short-term), stores (long-term), caches.

Three different things that people often mix up:

| Layer          | Scope                    | What it holds                              | Classes here                         |
|----------------|--------------------------|--------------------------------------------|--------------------------------------|
| Checkpointer   | one thread               | every superstep's state, pending writes, interrupts | `InMemorySaver`, `SqliteSaver`, `AsyncSqliteSaver` |
| Store          | across threads and users | long-term memory, searchable by namespace, filter and meaning | `InMemoryStore`, `SqliteStore`, `AsyncSqliteStore` |
| Cache          | across threads           | results of nodes/tasks with a `CachePolicy` | `InMemoryCache`, `SqliteCache`       |

Production rules encoded below:

* **Serialization is an allowlist.** `JsonPlusSerializer` warns today, and
  will refuse in a future version, when a checkpoint contains a class that
  was not explicitly allowed. List your custom state types.
* **Encrypt checkpoints at rest** when they contain user data:
  `EncryptedSerializer` wraps the serializer with AES (key from config).
* **TTL is a store feature, not a `put` feature.** `InMemoryStore` raises on
  `ttl=`; `SqliteStore(ttl=TTLConfig(...))` supports it.
* **Semantic search needs an index.** `IndexConfig(dims, embed, fields)` tells
  the store which JSON paths of each item to embed.
* In production, swap SQLite for Postgres (`langgraph-checkpoint-postgres`);
  the interfaces are identical.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from langgraph.cache.base import BaseCache
from langgraph.cache.memory import InMemoryCache
from langgraph.cache.sqlite import SqliteCache
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.base import BaseStore, Embeddings, IndexConfig, TTLConfig
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore
from langgraph.types import Checkpointer, ensure_valid_checkpointer

from lg_prod.config import Settings

# Custom classes that may appear in checkpointed state.
ALLOWED_STATE_TYPES: list[tuple[str, ...]] = [("lg_prod.models", "Classification")]


class HashingEmbeddings(Embeddings):
    """Deterministic, offline bag-of-words embeddings (feature hashing).

    Good enough to demonstrate and test semantic search without an API key.
    In production use a real model, e.g. `init_embeddings("openai:...")`.
    """

    def __init__(self, dims: int = 64) -> None:
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def incident_index(dims: int = 64) -> IndexConfig:
    """Embed only the `summary` field of stored incidents."""
    return IndexConfig(dims=dims, embed=HashingEmbeddings(dims), fields=["summary"])


def build_serializer(settings: Settings) -> SerializerProtocol:
    serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_STATE_TYPES)
    if settings.checkpoint_aes_key is None:
        return serde
    key = settings.checkpoint_aes_key.get_secret_value().encode()
    return EncryptedSerializer.from_pycryptodome_aes(serde, key=key)


def validate_checkpointer(checkpointer: Any) -> Checkpointer:
    """Fail fast on a mis-wired dependency (e.g. a saver *class* instead of an instance)."""
    return ensure_valid_checkpointer(checkpointer)


@dataclass
class Persistence:
    checkpointer: BaseCheckpointSaver[Any]
    store: BaseStore
    cache: BaseCache[Any]


@asynccontextmanager
async def open_persistence(settings: Settings) -> AsyncIterator[Persistence]:
    """Build all three layers from settings; closes connections on exit.

    The incident desk has async nodes, so it runs with `ainvoke`/`astream`.
    `SqliteSaver` is sync-only (its async methods raise NotImplementedError),
    so the SQLite flavour uses `AsyncSqliteSaver`. Use `SqliteSaver` only for
    graphs you drive with sync `invoke`/`stream`.
    """
    serde = build_serializer(settings)
    if settings.persistence == "memory":
        yield Persistence(
            checkpointer=InMemorySaver(serde=serde),
            store=InMemoryStore(index=incident_index()),
            cache=InMemoryCache(),
        )
        return
    # `check_same_thread=False`: sync nodes (and their store calls) run in a
    # thread pool, not on the thread that opened the connection.
    # Separate files: the async saver and the store write concurrently, and two
    # writers on one SQLite file fail with "database is locked".
    store_conn = sqlite3.connect(
        settings.sqlite_path + ".store", check_same_thread=False, isolation_level=None
    )
    try:
        store = SqliteStore(
            store_conn,
            index=incident_index(),
            ttl=TTLConfig(
                default_ttl=settings.memory_ttl_minutes,
                refresh_on_read=True,
                sweep_interval_minutes=60,
            ),
        )
        store.setup()
        async with AsyncSqliteSaver.from_conn_string(settings.sqlite_path) as saver:
            saver.serde = serde
            yield Persistence(
                checkpointer=saver,
                store=store,
                cache=SqliteCache(path=settings.sqlite_path + ".cache"),
            )
    finally:
        store_conn.close()
