"""The incident desk: the main `StateGraph`, one node per production concern.

Flow::

    START -> intake -> classify -> fetch_metrics --(Send x N)--> run_check
          run_check -> assess (defer) --Command--> plan_remediation | escalate | close
          plan_remediation -> approve --(interrupt)--> remediate | escalate
          remediate -> close ;  escalate (subgraph) --Command.PARENT--> close
          close -> remember -> END            (built with `add_sequence`)

What each node teaches:

* `intake`      - node-level `input_schema`, `Runtime` (context + store),
                  `get_config()`, a custom stream event, `push_ui_message`,
                  `UntrackedValue`, `TracePolicy(process_inputs=omit_payload)`.
* `classify`    - `CachePolicy` (needs `compile(cache=...)`), and an
                  `error_handler` that falls back to rules if the LLM fails.
* `fetch_metrics` - async node with `TimeoutPolicy` + `RetryPolicy`
                  (timeouts are async-only) and a degrade-gracefully handler.
* `run_check`   - map-reduce via `Send`, `ExecutionInfo.node_attempt` to
                  degrade on the final retry, `get_stream_writer()`.
* `assess`      - `defer=True` (waits for the whole fan-out), managed values
                  `RemainingSteps`/`IsLastStep`, `Command` routing with a
                  `Literal` return type, an `EphemeralValue` hand-off note.
* `approve`     - human-in-the-loop `interrupt()` with a validated
                  `response_schema`, authorization from `runtime.context`.
* `escalate`    - a compiled subgraph used as a node (`escalation.py`).
* `close` / `remember` - `add_sequence`, `Overwrite` to compact a reducer
                  channel, `push_message`, long-term memory via `get_store()`.

Why one big graph? It mirrors a real on-call workflow, and every step is
either deterministic or behind an injected interface, so the whole thing is
testable offline.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.cache.base import BaseCache
from langgraph.config import get_config, get_store, get_stream_writer
from langgraph.errors import NodeError, NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import push_message
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.ui import RemoveUIMessage, UIMessage, delete_ui_message, push_ui_message
from langgraph.runtime import Runtime, get_runtime
from langgraph.store.base import BaseStore
from langgraph.types import (
    All,
    CachePolicy,
    Checkpointer,
    Command,
    Overwrite,
    RetryPolicy,
    Send,
    TimeoutPolicy,
    TracePolicy,
    interrupt,
    omit_payload,
)

from lg_prod.config import Settings
from lg_prod.context import DeskContext
from lg_prod.graphs.escalation import build_escalation_graph
from lg_prod.models import Classifier, RuleClassifier
from lg_prod.services import PermanentBackendError, Services, TransientBackendError
from lg_prod.state import (
    SEVERITY_RANK,
    CheckInput,
    Finding,
    IncidentInput,
    IncidentOutput,
    IncidentState,
    Scratchpad,
    Severity,
)

IncidentGraph = CompiledStateGraph[IncidentState, DeskContext, IncidentInput, IncidentOutput]

CHECKS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "latency": ("db", "cache", "network"),
    "errors": ("db", "deploys", "dependencies"),
    "capacity": ("cpu", "memory", "disk"),
    "security": ("auth_logs", "waf"),
    "unknown": ("db", "network"),
}
RUNBOOK: dict[str, str] = {
    "db": "failover_db",
    "cache": "flush_cache",
    "cpu": "scale_out",
    "memory": "restart_pods",
    "deploys": "rollback_deploy",
}
TIMELINE_KEEP = 20


class RemediationInput(Scratchpad):
    """What the remediation nodes read, including the private Scratchpad keys.

    A node only receives keys in its input schema. The default is the graph's
    state schema, which deliberately lacks the private keys, so every node
    that reads `runbook` must declare an input schema like this one.
    """

    incident_id: str
    service: str
    severity: Severity
    findings: list[Finding]


class NodeUpdate(IncidentState, Scratchpad, total=False):
    """Return type for nodes that write both shared and private keys."""


class ApprovalDecision(TypedDict):
    """Resume value for the `approve` interrupt. Validated by `interrupt()`."""

    approved: bool
    note: str


def build_incident_graph(
    settings: Settings,
    services: Services,
    *,
    classifier: Classifier | None = None,
    checkpointer: Checkpointer = None,
    store: BaseStore | None = None,
    cache: BaseCache[object] | None = None,
    interrupt_before: All | list[str] | None = None,
    interrupt_after: All | list[str] | None = None,
) -> IncidentGraph:
    """Assemble and compile the incident desk. Every collaborator is injectable."""
    classifier = classifier or RuleClassifier()
    fallback = RuleClassifier()

    def needs_approval(severity: Severity) -> bool:
        return SEVERITY_RANK[severity] < SEVERITY_RANK[settings.auto_remediate_max_severity]

    # ------------------------------------------------------------------ nodes

    def intake(state: IncidentInput, runtime: Runtime[DeskContext]) -> NodeUpdate:
        # `input_schema=IncidentInput` on add_node: this node sees only those keys.
        thread_id = get_config()["configurable"].get("thread_id", "adhoc")
        incident_id = f"INC-{str(thread_id)[:8]}"
        similar: list[str] = []
        if runtime.store is not None:
            # Semantic ranking *within* this service: `filter` is an exact
            # match on stored fields, `query` ranks by embedding similarity.
            hits = runtime.store.search(
                runtime.context.memory_namespace,
                query=state["report"],
                filter={"service": state["service"]},
                limit=3,
            )
            similar = [hit.value["summary"] for hit in hits]
        runtime.stream_writer({"event": "intake", "incident_id": incident_id})
        push_ui_message(
            "incident_card",
            {"incident_id": incident_id, "service": state["service"], "status": "open"},
            id=f"card-{incident_id}",
        )
        return {
            "incident_id": incident_id,
            "status": "open",
            "messages": [HumanMessage(state["report"], name=runtime.context.user_id)],
            "timeline": [f"opened by {runtime.context.user_id}"],
            "audit": [f"intake:{runtime.context.user_id}"],
            "trace_url": f"https://traces.example/{incident_id}",
            "similar_incidents": similar,  # private key (Scratchpad)
        }

    def classify(state: IncidentState) -> IncidentState:
        result = classifier.classify(state["report"], state["service"])
        return {
            "severity": result.severity,
            "category": result.category,
            "timeline": [f"classified {result.severity}/{result.category} ({result.rationale})"],
        }

    def classify_fallback(
        state: IncidentState, error: NodeError
    ) -> Command[Literal["fetch_metrics"]]:
        # Error handlers are separate nodes: a plain dict return would NOT
        # follow `classify`'s outgoing edge, so route explicitly with Command.
        result = fallback.classify(state["report"], state["service"])
        return Command(
            goto="fetch_metrics",
            update={
                "severity": result.severity,
                "category": result.category,
                "timeline": [f"classifier failed ({type(error.error).__name__}); used rules"],
            },
        )

    async def fetch_metrics(state: IncidentState) -> IncidentState:
        metrics = await services.metrics.fetch(state["service"])
        detail = ", ".join(f"{k}={v:g}" for k, v in sorted(metrics.items()))
        return {"timeline": [f"metrics: {detail}"]}

    def metrics_unavailable(
        state: IncidentState, error: NodeError
    ) -> Command[Literal["run_check"]]:
        reason = "timed out" if isinstance(error.error, NodeTimeoutError) else str(error.error)
        return Command(
            goto=[
                Send("run_check", CheckInput(service=state["service"], check=c))
                for c in CHECKS_BY_CATEGORY[state.get("category", "unknown")]
            ],
            update={"timeline": [f"metrics unavailable: {reason}"]},
        )

    def plan_diagnostics(state: IncidentState) -> list[Send]:
        # Map step of map-reduce: one `Send` per check, each with its own input.
        return [
            Send("run_check", CheckInput(service=state["service"], check=check))
            for check in CHECKS_BY_CATEGORY[state.get("category", "unknown")]
        ]

    def run_check(state: CheckInput, runtime: Runtime[DeskContext]) -> IncidentState:
        writer = get_stream_writer()
        attempt = runtime.execution_info.node_attempt if runtime.execution_info else 1
        try:
            finding = services.diagnostics.run(state["service"], state["check"])
        except TransientBackendError as exc:
            # Fan-out siblings share a superstep, where node `error_handler`s
            # do not rescue the run (see tests). Retry via RetryPolicy, and on
            # the last attempt degrade to a failed finding instead of raising.
            if attempt < settings.node_max_attempts:
                raise
            finding = {"check": state["check"], "ok": False, "detail": f"probe error: {exc}"}
        writer({"event": "check", "check": state["check"], "ok": finding["ok"]})
        return {"findings": [finding]}

    def assess(state: IncidentState) -> Command[Literal["plan_remediation", "escalate", "close"]]:
        failing = [f for f in state.get("findings", []) if not f["ok"]]
        summary = (
            f"{len(failing)} failing check(s): " + ", ".join(f["check"] for f in failing)
            if failing
            else "all checks healthy"
        )
        update: IncidentState = {"summary": summary, "timeline": [f"assessed: {summary}"]}
        # Managed values: if the run is about to hit `recursion_limit`, stop
        # doing work and escalate to a human rather than crash.
        if state.get("is_last_step") or state.get("remaining_steps", 99) < 6:
            return Command(goto="escalate", update={**update, "handoff_note": "step budget"})
        if not failing:
            return Command(goto="close", update={**update, "status": "closed"})
        if any(f["check"] in RUNBOOK for f in failing):
            return Command(goto="plan_remediation", update=update)
        return Command(goto="escalate", update={**update, "handoff_note": "no runbook"})

    def plan_remediation(state: RemediationInput) -> NodeUpdate:
        failing = [f["check"] for f in state["findings"] if not f["ok"]]
        action = next(RUNBOOK[c] for c in failing if c in RUNBOOK)
        hint = (
            f" (seen before: {state['similar_incidents'][0]})"
            if state.get("similar_incidents")
            else ""
        )
        if needs_approval(state["severity"]):
            card: UIMessage = push_ui_message(
                "approval_request",
                {"action": action, "severity": state["severity"]},
                id=f"approval-{state['incident_id']}",
            )
            get_stream_writer()({"event": "awaiting_approval", "card": card["id"]})
        return {"runbook": action, "timeline": [f"planned {action}{hint}"]}

    def approve(
        state: RemediationInput, runtime: Runtime[DeskContext]
    ) -> Command[Literal["remediate", "escalate"]]:
        severity = state["severity"]
        if not needs_approval(severity):
            return Command(goto="remediate", update={"approved_by": "policy"})
        # Everything before `interrupt()` re-runs on resume, and any state
        # writes made before it are discarded when the node pauses. That is why
        # the approval card is pushed by `plan_remediation`, not here.
        decision = interrupt(
            {
                "type": "approval",
                "incident_id": state.get("incident_id"),
                "action": state.get("runbook"),
                "severity": severity,
            },
            response_schema=ApprovalDecision,
        )
        # Authorization uses the *resuming* caller's context, not stored state.
        if decision["approved"] and runtime.context.can_approve:
            return Command(goto="remediate", update={"approved_by": runtime.context.user_id})
        reason = decision["note"] if not decision["approved"] else "approver lacks role"
        return Command(
            goto="escalate",
            update={"status": "rejected", "timeline": [f"remediation rejected: {reason}"]},
        )

    def remediate(state: RemediationInput) -> IncidentState:
        result = services.remediator.apply(state["service"], state["runbook"])
        return {"status": "remediated", "actions_taken": [result], "timeline": [result]}

    def close(state: IncidentState) -> dict[str, Any]:
        status = state.get("status", "open")
        final = "closed" if status in ("open", "closed") else status
        text = f"{state.get('incident_id')} {final}: {state.get('summary', '')}"
        push_message(AIMessage(text, id=f"summary-{state.get('incident_id')}"))
        push_ui_message(
            "incident_card",
            {"status": final},
            id=f"card-{state.get('incident_id')}",
            merge=True,
        )
        # The approval card (if any) is obsolete once the incident closes.
        if any(m["id"] == f"approval-{state.get('incident_id')}" for m in state.get("ui", [])):
            removed: RemoveUIMessage = delete_ui_message(f"approval-{state.get('incident_id')}")
            get_stream_writer()({"event": "ui_removed", "id": removed["id"]})
        timeline = state.get("timeline", [])
        return {
            "status": final,
            # Bypass the `operator.add` reducer once to compact the channel.
            "timeline": Overwrite([*timeline[-TIMELINE_KEEP:], f"closed as {final}"]),
            "audit": ["close"],
        }

    def remember(state: IncidentState) -> IncidentState:
        # `get_store()` / `get_runtime()` reach the same objects as an injected
        # `runtime` argument. Handy in helpers that don't receive `runtime`.
        store = get_store()
        ctx = get_runtime(DeskContext).context
        if store is not None:
            store.put(
                ctx.memory_namespace,
                state.get("incident_id", str(uuid.uuid4())),
                {
                    "service": state["service"],
                    "summary": f"{state['service']}: {state.get('summary', '')}",
                    "actions": state.get("actions_taken", []),
                    "closed_at": time.time(),
                },
                # InMemoryStore raises on `ttl=`; only TTL-capable stores
                # (SqliteStore, Postgres) advertise `supports_ttl`.
                ttl=settings.memory_ttl_minutes if store.supports_ttl else None,
            )
        return {"audit": ["remember"]}

    # ------------------------------------------------------------------ wiring

    transient = RetryPolicy(
        max_attempts=settings.node_max_attempts,
        initial_interval=0.01,
        backoff_factor=2.0,
        jitter=False,
        retry_on=TransientBackendError,
    )
    builder = StateGraph(
        IncidentState,
        context_schema=DeskContext,
        input_schema=IncidentInput,
        output_schema=IncidentOutput,
    )
    # Defaults for every node; per-node arguments override them.
    builder.set_node_defaults(retry_policy=transient)

    builder.add_node(
        "intake",
        intake,
        input_schema=IncidentInput,
        trace_policy=TracePolicy(process_inputs=omit_payload),
        metadata={"owner": "sre", "pii": True},
    )
    builder.add_node(
        "classify",
        classify,
        cache_policy=CachePolicy(
            # Must tolerate missing keys: `get_graph()` dry-runs task
            # preparation with an empty state and calls key_func on it.
            key_func=lambda s: f"{s.get('service')}|{s.get('report')}",
            ttl=settings.classification_cache_ttl_s,
        ),
        error_handler=classify_fallback,
    )
    builder.add_node(
        "fetch_metrics",
        fetch_metrics,
        timeout=TimeoutPolicy(run_timeout=settings.metrics_timeout_s),
        retry_policy=[
            # The first matching policy wins: never retry permanent errors.
            RetryPolicy(max_attempts=1, retry_on=PermanentBackendError),
            RetryPolicy(
                max_attempts=settings.node_max_attempts,
                initial_interval=0.01,
                jitter=False,
                retry_on=(TransientBackendError, NodeTimeoutError),
            ),
        ],
        error_handler=metrics_unavailable,
    )
    builder.add_node("run_check", run_check, input_schema=CheckInput)
    builder.add_node("assess", assess, defer=True)
    builder.add_node("plan_remediation", plan_remediation, input_schema=RemediationInput)
    builder.add_node("approve", approve, input_schema=RemediationInput)
    builder.add_node("remediate", remediate, input_schema=RemediationInput)
    builder.add_node(
        "escalate",
        build_escalation_graph(services),
        destinations=("close",),  # rendering hint: the subgraph jumps to `close`
    )
    builder.add_sequence([("close", close), ("remember", remember)])

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "classify")
    builder.add_edge("classify", "fetch_metrics")
    builder.add_conditional_edges("fetch_metrics", plan_diagnostics, ["run_check"])
    builder.add_edge("run_check", "assess")
    builder.add_edge("plan_remediation", "approve")
    builder.add_edge("remediate", "close")
    builder.add_edge("remember", END)

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
        cache=cache,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        name="incident_desk",
    )
