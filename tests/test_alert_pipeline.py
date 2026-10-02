"""Low-level Pregel: NodeBuilder, every channel type, a custom channel."""

from __future__ import annotations

import pytest
from langgraph.channels import LastValue
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import EmptyChannelError, InvalidUpdateError
from langgraph.pregel import NodeBuilder, Pregel

from lg_prod.graphs.alert_pipeline import MaxValue, build_alert_pipeline
from tests.conftest import thread

BATCH = [
    {"id": "a1", "source": "prom", "level": "warning"},
    {"id": "a2", "source": "prom", "level": "critical"},
]


def test_pipeline_output_and_superstep_order():
    pipeline = build_alert_pipeline()

    order = [next(iter(update)) for update in pipeline.stream(BATCH, stream_mode="updates")]
    out = pipeline.invoke(BATCH)

    assert out == {
        "digest": "2 unique alert(s), peak=1 -> page",
        "published": "2 UNIQUE ALERT(S), PEAK=1 -> PAGE",
        "count": 2,
        "archived": 2,
        # `status` (AnyValue) was cleared in a later step without writes.
    }
    # dedupe+score in parallel, then the barrier fires correlate once, then
    # the ephemeral tick fires notify (TAG_HIDDEN: not streamed);
    # *AfterFinish subscribers run last.
    assert sorted(order[:2]) == ["dedupe", "score"]
    assert order[2] == "correlate"
    assert "notify" not in order
    assert sorted(order[3:]) == ["archive", "publish"]


def test_accumulating_topic_dedupes_across_invocations():
    pipeline = build_alert_pipeline(InMemorySaver())
    cfg = thread("p1")

    pipeline.invoke(BATCH, cfg)
    out = pipeline.invoke([*BATCH, {"id": "a3", "source": "x", "level": "info"}], cfg)

    assert out["count"] == 3  # a1/a2 were in `seen` (Topic accumulate=True)
    state = pipeline.get_state(cfg)
    assert "debug_scratch" not in state.values  # UntrackedValue never persisted


def test_custom_channel_contract():
    channel = MaxValue().from_checkpoint(None)
    with pytest.raises(EmptyChannelError):
        channel.get()
    assert channel.update([0.2, 0.7]) is True
    assert channel.update([0.5]) is False  # no change -> subscribers not triggered
    restored = MaxValue().from_checkpoint(channel.checkpoint())
    assert restored.get() == 0.7
    with pytest.raises(InvalidUpdateError, match="INVALID_GRAPH_NODE_RETURN_VALUE"):
        channel.update([-1.0])


def test_minimal_pregel_app():
    app = Pregel(
        nodes={"double": NodeBuilder().subscribe_only("x").do(lambda x: x * 2).write_to("y")},
        channels={"x": LastValue(int), "y": LastValue(int)},
        input_channels="x",
        output_channels="y",
    )
    assert app.invoke(21) == 42
