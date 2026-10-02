"""Interactive demo: `uv run python -m lg_prod [--llm]`.

Runs the incident desk against the in-memory demo backends. With `--llm`
the classifier uses `LG_CHAT_MODEL` (needs the provider's API key);
otherwise everything is offline and deterministic.

Try: `checkout returning 500 errors for customers` (service `checkout`) for an
approval, or `auth is down, total outage` (service `auth`) for an escalation.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid

from lg_prod.config import get_settings
from lg_prod.context import DeskContext
from lg_prod.graphs.triage import build_incident_graph
from lg_prod.models import LLMClassifier, build_chat_model
from lg_prod.persistence import open_persistence
from lg_prod.service import DeskReply, IncidentDeskService
from lg_prod.services import FakeDiagnostics, demo_services


def _ask_answer(reply: DeskReply) -> dict[str, object]:
    answers: dict[str, object] = {}
    for q in reply.questions:
        print(f"\n[{q.kind}] {json.dumps(q.payload)}")
        if q.kind == "approval":
            ok = input("approve? [y/N] ").strip().lower() == "y"
            answers[q.id] = {"approved": ok, "note": "via CLI"}
        else:
            answers[q.id] = {"acknowledged_by": input("your name: ").strip() or "cli"}
    return answers


async def main() -> None:
    settings = get_settings()
    services = demo_services()
    services.diagnostics = FakeDiagnostics(
        broken={("checkout", "db"), ("search", "cpu"), ("auth", "dependencies")}
    )
    classifier = LLMClassifier(build_chat_model(settings)) if "--llm" in sys.argv else None
    ctx = DeskContext(user_id="u-cli", role="oncall_lead")
    async with open_persistence(settings) as p:
        desk = build_incident_graph(
            settings,
            services,
            classifier=classifier,
            checkpointer=p.checkpointer,
            store=p.store,
            cache=p.cache,
        )
        svc = IncidentDeskService(desk, settings)
        print("Incident desk (on-call lead mode). Ctrl-D to quit.")
        while True:
            try:
                report = (await asyncio.to_thread(input, "\nreport> ")).strip()
                service = (
                    await asyncio.to_thread(input, "service [checkout/search/auth]> ")
                ).strip() or "checkout"
            except EOFError:
                break
            if not report:
                continue
            thread_id = str(uuid.uuid4())
            reply = await svc.report(thread_id, report, service, ctx)
            while reply.status == "awaiting_input":
                reply = await svc.answer(thread_id, ctx, _ask_answer(reply))
            print(f"\n=> {json.dumps(reply.output, indent=2)}")
            for line in (await svc.snapshot(thread_id)).values.get("timeline", []):
                print(f"   - {line}")


if __name__ == "__main__":
    asyncio.run(main())
