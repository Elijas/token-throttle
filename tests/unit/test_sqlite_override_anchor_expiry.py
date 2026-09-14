from __future__ import annotations

from contextlib import closing

import pytest

from token_throttle import frozen_usage
from token_throttle._limiter_backends._sqlite._engine import BucketSpec, SqliteEngine


def _engine(path, configured_limit):
    return SqliteEngine(
        db_path=str(path),
        key_prefix="anchor-expiry",
        model_family="shared",
        buckets=(BucketSpec("requests", 10, configured_limit),),
        bucket_ttl_seconds=100,
        override_ttl_seconds=2,
        refund_dedup_ttl_seconds=100,
        max_reservation_lifetime_seconds=10,
    )


@pytest.mark.parametrize(
    ("configured_limit", "now", "expected"),
    [
        (10, 101, 0.1),
        (10, 102, 0.2),
        (10, 105, 3.2),
        (20, 101, 2.0),
        (20, 102, 4.0),
        (20, 105, 10.0),
    ],
)
def test_override_expiry_uses_only_matching_configuration_history(
    tmp_path, configured_limit, now, expected
):
    path = tmp_path / "anchor.sqlite3"
    with closing(_engine(path, 10)) as writer:
        writer.consume(
            frozen_usage({"requests": 10}),
            reservation_id=None,
            reservation_lifetime_seconds=None,
            clock=lambda: 100.0,
        )
        writer.set_max_capacity("requests", 10, 1, clock=lambda: 100.0)

    with closing(_engine(path, configured_limit)) as reader:
        before = reader._connection.execute("SELECT * FROM buckets").fetchall()
        snapshots, _ = reader.inspect_snapshot(clock=lambda: now)
        assert snapshots[0].current_capacity == pytest.approx(expected)
        assert snapshots[0].override_active is (configured_limit == 10 and now < 102)
        assert reader._connection.execute("SELECT * FROM buckets").fetchall() == before
        result = reader.consume(
            frozen_usage({"requests": 0}),
            reservation_id=None,
            reservation_lifetime_seconds=None,
            clock=lambda: now,
        )
        assert result.pre_capacities[("requests", 10)] == pytest.approx(expected)
        assert result.post_capacities[("requests", 10)] == pytest.approx(expected)

    with closing(_engine(path, configured_limit)) as reopened:
        snapshots, _ = reopened.inspect_snapshot(clock=lambda: now)
        assert snapshots[0].current_capacity == pytest.approx(expected)
