"""Durable expiry boundaries for Redis runtime overrides."""

import json
import math
from dataclasses import dataclass
from decimal import Decimal, localcontext

from token_throttle._capacity import (
    CalculatedCapacity,
    _validate_max_capacity_finite_positive,
    calculate_capacity,
)

# Keep the established override payload format. Shared operation requires
# clients that maintain expiry history; older clients can invalidate it.
# The sidecar outlives the override, but never supplies missing bucket state.
WRITE_OVERRIDE = """
local previous = redis.call('GET', KEYS[2])
local ttl = redis.call('PTTL', KEYS[2])
if ARGV[1] == 'refresh' then
    local live = redis.call('GET', KEYS[1])
    if not live then return 0 end
    local ok, decoded = pcall(cjson.decode, live)
    if not ok or type(decoded) ~= 'table'
        or type(decoded.configured_max_capacity) ~= 'number'
        or decoded.override_max_capacity ~= tonumber(ARGV[6]) then return 0 end
    local configured = tonumber(ARGV[5])
    if math.abs(decoded.configured_max_capacity - configured) > 1e-12 *
        math.max(math.abs(decoded.configured_max_capacity), math.abs(configured))
        then return 0 end
end
local history_ttl = math.max(tonumber(ARGV[4]) * 1000, ttl,
    redis.call('PTTL', KEYS[3]), redis.call('PTTL', KEYS[4]))
local saved = redis.pcall('SET', KEYS[2], ARGV[3], 'PX', history_ttl)
if type(saved) == 'table' and saved.err then return saved end
local written
if ARGV[1] == 'refresh' then
    written = redis.pcall('EXPIRE', KEYS[1], ARGV[7])
else
    written = redis.pcall('SET', KEYS[1], ARGV[2], 'EX', ARGV[7])
end
if type(written) == 'table' and written.err then
    local restored
    if previous and ttl > 0 then
        restored = redis.pcall('SET', KEYS[2], previous, 'PX', ttl)
    elseif previous and ttl == -1 then
        restored = redis.pcall('SET', KEYS[2], previous)
    else
        restored = redis.pcall('DEL', KEYS[2])
    end
    if type(restored) == 'table' and restored.err then return restored end
    return written
end
return 1
"""

RETAIN_EXPIRY = """
local ttl = redis.call('PTTL', KEYS[1])
if ttl == -2 then return 0 end
return redis.call('PEXPIRE', KEYS[1], math.max(tonumber(ARGV[1]) * 1000,
    ttl, redis.call('PTTL', KEYS[2]), redis.call('PTTL', KEYS[3])))
"""


@dataclass(frozen=True)
class OverrideExpiry:
    configured: float
    maximum: float
    expires_at: float


def parse_expiry(raw: bytes | str | None) -> OverrideExpiry | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
        if (
            not isinstance(value, dict)
            or type(value.get("version")) is not int
            or value["version"] != 1
        ):
            raise ValueError("Unsupported Redis override expiry history")
        configured = _validate_max_capacity_finite_positive(value["configured"])
        maximum = _validate_max_capacity_finite_positive(value["maximum"])
        expiry = value["expires_at"]
        if type(expiry) not in (int, float) or not math.isfinite(expiry) or expiry < 0:
            raise ValueError("Invalid Redis override expiry boundary")
        return OverrideExpiry(configured, maximum, float(expiry))
    except (KeyError, TypeError, UnicodeError, OverflowError) as exc:
        raise ValueError("Invalid Redis override expiry history") from exc


def expiry_payload(configured: float, maximum: float, expires_at: float) -> str:
    return json.dumps(
        {
            "version": 1,
            "configured": configured,
            "maximum": maximum,
            "expires_at": expires_at,
        }
    )


def accrue_with_expiry(  # noqa: PLR0913
    history: OverrideExpiry | None,
    *,
    override: float | None,
    configured: float,
    per_seconds: int,
    last_checked: float,
    current_time: float,
    stored: float,
    rate_per_sec: float,
) -> float:
    """Integrate validated state without capping intermediate balances."""
    if (
        history is None
        or override is not None
        or not math.isclose(history.configured, configured, rel_tol=1e-12)
        or current_time < history.expires_at
    ):
        return stored + max(0.0, current_time - last_checked) * rate_per_sec
    old_seconds = max(0.0, history.expires_at - last_checked)
    new_seconds = max(0.0, current_time - max(last_checked, history.expires_at))
    old_rate = history.maximum / per_seconds
    terms = (stored, old_seconds * old_rate, new_seconds * rate_per_sec)
    if all(math.isfinite(term) for term in terms):
        try:
            return math.fsum(terms)
        except OverflowError:
            pass
    # Float products can overflow before cancelling finite debt. Use exact
    # float operands at sufficient precision only on this exceptional path.
    with localcontext() as ctx:
        ctx.prec = 2048
        return float(
            Decimal(stored)
            + Decimal(old_seconds) * Decimal(old_rate)
            + Decimal(new_seconds) * Decimal(rate_per_sec)
        )


def calculate_with_expiry(  # noqa: PLR0913
    *,
    history: OverrideExpiry | None,
    override: float | None,
    configured: float,
    per_seconds: int,
    last_checked: float | str | bytes | None,
    outdated_capacity: float | str | bytes | None,
    current_time: float,
    bucket_id: str,
) -> CalculatedCapacity:
    maximum = configured if override is None else override
    # Validate original values before integrating individual rate intervals.
    result = calculate_capacity(
        last_checked,
        outdated_capacity,
        current_time,
        maximum,
        maximum / per_seconds,
        bucket_id,
    )
    if last_checked is None or outdated_capacity is None:
        return result
    accrued = accrue_with_expiry(
        history,
        override=override,
        configured=configured,
        per_seconds=per_seconds,
        last_checked=float(last_checked),
        current_time=float(current_time),
        stored=float(outdated_capacity),
        rate_per_sec=maximum / per_seconds,
    )
    return CalculatedCapacity(amount=min(maximum, accrued), is_fresh_start=False)
