"""Backends the graphs talk to, behind Protocols.

Graph factories take a `Services` bundle instead of importing clients, so the
same graph runs against real systems in production and against these
in-memory fakes in tests and demos. Nodes never construct clients themselves.

Two error types make retry policy explicit:

* `TransientBackendError` - worth retrying (`RetryPolicy(retry_on=...)`).
* `PermanentBackendError` - retrying cannot help; handle or surface it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol

from lg_prod.state import Finding


class TransientBackendError(Exception):
    """Timeouts, 5xx, connection resets. Safe to retry."""


class PermanentBackendError(Exception):
    """Bad request, unknown service, permission denied. Do not retry."""


class MetricsBackend(Protocol):
    async def fetch(self, service: str) -> dict[str, float]: ...


class DiagnosticsBackend(Protocol):
    def run(self, service: str, check: str) -> Finding: ...


class Pager(Protocol):
    def page(self, team: str, message: str) -> str: ...


class Remediator(Protocol):
    def apply(self, service: str, action: str) -> str: ...


@dataclass
class Services:
    metrics: MetricsBackend
    diagnostics: DiagnosticsBackend
    pager: Pager
    remediator: Remediator


# --- In-memory implementations ---------------------------------------------


@dataclass
class FakeMetrics:
    data: dict[str, dict[str, float]] = field(default_factory=dict)
    delay_s: float = 0.0
    failures_before_success: int = 0
    calls: int = 0

    async def fetch(self, service: str) -> dict[str, float]:
        self.calls += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.calls <= self.failures_before_success:
            raise TransientBackendError(f"metrics API 503 (call {self.calls})")
        if service not in self.data:
            raise PermanentBackendError(f"unknown service {service!r}")
        return self.data[service]


@dataclass
class FakeDiagnostics:
    broken: set[tuple[str, str]] = field(default_factory=set)
    crashing: set[str] = field(default_factory=set)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def run(self, service: str, check: str) -> Finding:
        self.calls.append((service, check))
        if check in self.crashing:
            raise TransientBackendError(f"{check} probe crashed")
        ok = (service, check) not in self.broken
        return {"check": check, "ok": ok, "detail": "healthy" if ok else f"{check} failing"}


@dataclass
class FakePager:
    pages: list[tuple[str, str]] = field(default_factory=list)

    def page(self, team: str, message: str) -> str:
        self.pages.append((team, message))
        return f"PAGE-{len(self.pages)}"


@dataclass
class FakeRemediator:
    applied: list[tuple[str, str]] = field(default_factory=list)

    def apply(self, service: str, action: str) -> str:
        self.applied.append((service, action))
        return f"{action} applied to {service}"


def demo_services() -> Services:
    return Services(
        metrics=FakeMetrics(
            data={
                "checkout": {"p99_ms": 2300.0, "error_rate": 0.12, "cpu": 0.55},
                "search": {"p99_ms": 180.0, "error_rate": 0.001, "cpu": 0.93},
                "auth": {"p99_ms": 90.0, "error_rate": 0.0, "cpu": 0.20},
            }
        ),
        diagnostics=FakeDiagnostics(broken={("checkout", "db"), ("search", "cpu")}),
        pager=FakePager(),
        remediator=FakeRemediator(),
    )
