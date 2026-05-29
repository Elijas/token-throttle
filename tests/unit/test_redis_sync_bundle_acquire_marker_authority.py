"""Sync Redis marker-authority focused test entrypoint.

The sync Redis marker-authority cases live with the async parity cases in
``test_bundle_acquire_marker_authority``. This module keeps the focused Redis
sync gate addressable by filename.
"""

from tests.unit import test_bundle_acquire_marker_authority as marker_authority_tests

test_sync_redis_happy_path_deletes_marker_and_writes_tombstone = marker_authority_tests.test_sync_redis_happy_path_deletes_marker_and_writes_tombstone
test_sync_redis_forged_reservation_is_unknown = (
    marker_authority_tests.test_sync_redis_forged_reservation_is_unknown
)
test_sync_redis_duplicate_refund_raises_duplicate = (
    marker_authority_tests.test_sync_redis_duplicate_refund_raises_duplicate
)
test_sync_redis_cross_limiter_refund_rejects_shared_marker = (
    marker_authority_tests.test_sync_redis_cross_limiter_refund_rejects_shared_marker
)
test_sync_redis_marker_ttl_expiry_is_unknown = (
    marker_authority_tests.test_sync_redis_marker_ttl_expiry_is_unknown
)
