"""Redis 6.2-compatible state TTL refresh without shortening debt retention."""

from dataclasses import dataclass
from typing import TypedDict


class RetentionKwargs(TypedDict, total=False):
    retention_max_capacity: float


@dataclass(frozen=True)
class BucketSnapshotPlan:
    current_time: float
    capacity: float | None
    max_capacity_override: float | None
    retention_kwargs: RetentionKwargs


# Each invocation keeps the existing four-result capacity-read pipeline shape.
# It never creates missing state or turns partial-state evidence into a pair.
# Using PTTL avoids losing a fractional second from an existing longer deadline.
REFRESH_STATE_TTL_SCRIPT = """
local ttl = tonumber(ARGV[1]) * 1000
local capacity = tonumber(redis.call('GET', KEYS[2]))
if capacity and capacity < 0 and capacity > -math.huge then
    local required = math.ceil((1 - capacity / tonumber(ARGV[2])) * tonumber(ARGV[3]))
    if required > 2147483647 then
        return redis.error_reply('Unpaid capacity debt exceeds finite retention limit')
    end
    ttl = math.max(ttl, required * 1000)
end
local remaining = redis.call('PTTL', KEYS[1])
if remaining == -2 then return 0 end
local history_remaining = redis.call('PTTL', KEYS[3])
local history_required = math.max(ttl, remaining)
if history_remaining >= 0 and history_remaining < history_required then
    redis.call('PEXPIRE', KEYS[3], history_required)
end
if remaining >= ttl then return 1 end
return redis.call('PEXPIRE', KEYS[1], ttl)
"""
