"""Regression coverage for the v1.4.x -> v2.0.0 config migration helper."""

from __future__ import annotations

from copy import deepcopy

import token_throttle
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
from token_throttle.migration import (
    ConfigMigrationIssue,
    validate_config_for_v2_0,
)


def _clean_config_dict() -> dict[str, object]:
    return {
        "quotas": [
            {"metric": "requests", "limit": 1000, "per_seconds": 60},
            {"metric": "tokens", "limit": 100_000.0, "per_seconds": 60},
        ],
        "model_family": "gpt-4o",
    }


def _issue_by_path(
    issues: list[ConfigMigrationIssue],
) -> dict[str, ConfigMigrationIssue]:
    return {issue.field_path: issue for issue in issues}


def test_clean_v2_config_dict_returns_empty_error_list() -> None:
    assert validate_config_for_v2_0(_clean_config_dict()) == []


def test_clean_v2_config_object_returns_empty_error_list() -> None:
    config = PerModelConfig(
        quotas=UsageQuotas([Quota(metric="requests", limit=1000, per_seconds=60)]),
        model_family="gpt-4o",
    )

    assert validate_config_for_v2_0(config) == []


def test_string_limit_reports_use_int_or_float() -> None:
    config = _clean_config_dict()
    config["quotas"][0]["limit"] = "1000"  # type: ignore[index]

    issue = _issue_by_path(validate_config_for_v2_0(config))["config.quotas[0].limit"]

    assert issue.value == "1000"
    assert "limit must be int or float, not str" in issue.reason
    assert issue.suggested_fix is not None
    assert "int or float" in issue.suggested_fix


def test_float_per_seconds_reports_use_int() -> None:
    config = _clean_config_dict()
    config["quotas"][0]["per_seconds"] = 60.0  # type: ignore[index]

    issue = _issue_by_path(validate_config_for_v2_0(config))[
        "config.quotas[0].per_seconds"
    ]

    assert issue.value == 60.0
    assert issue.reason == "per_seconds must be int, not float"
    assert issue.suggested_fix == "Use the integer 60."


def test_whitespace_colon_and_braces_report_separate_segment_errors() -> None:
    config = {
        "quotas": [
            {
                "metric": "bad metric:{hash}",
                "limit": 100,
                "per_seconds": 60,
            }
        ],
        "model_family": " model:{family} ",
    }

    issues = validate_config_for_v2_0(config)
    metric_reasons = [
        issue.reason
        for issue in issues
        if issue.field_path == "config.quotas[0].metric"
    ]
    family_reasons = [
        issue.reason for issue in issues if issue.field_path == "config.model_family"
    ]

    assert "metric must not contain whitespace" in metric_reasons
    assert "metric must not contain ':'" in metric_reasons
    assert "metric must not contain '{' or '}'" in metric_reasons
    assert "model_family must not contain leading/trailing whitespace" in family_reasons
    assert "model_family must not contain ':'" in family_reasons
    assert "model_family must not contain '{' or '}'" in family_reasons


def test_multiple_errors_are_reported_without_stopping_at_first() -> None:
    config = {
        "quotas": [
            {"metric": b"tokens", "limit": "1000", "per_seconds": 60.0},
            {"metric": "requests:per_minute", "limit": 100, "per_seconds": b"60"},
        ],
        "model_family": "gpt 4o",
    }

    issues = validate_config_for_v2_0(config)
    paths = [issue.field_path for issue in issues]

    assert "config.quotas[0].metric" in paths
    assert "config.quotas[0].limit" in paths
    assert "config.quotas[0].per_seconds" in paths
    assert "config.quotas[1].metric" in paths
    assert "config.quotas[1].per_seconds" in paths
    assert "config.model_family" in paths
    assert len(issues) >= 6


def test_helper_does_not_mutate_input() -> None:
    config = {
        "quotas": [
            {"metric": " tokens ", "limit": "1000", "per_seconds": 60.0},
        ],
        "model_family": "gpt 4o",
    }
    original = deepcopy(config)

    validate_config_for_v2_0(config)

    assert config == original


def test_public_api_imports_work() -> None:
    assert token_throttle.validate_config_for_v2_0 is validate_config_for_v2_0
    assert token_throttle.ConfigMigrationIssue is ConfigMigrationIssue


def test_redis_builder_config_missing_key_prefix_reports_upgrade_step() -> None:
    issues = validate_config_for_v2_0({"backend": "redis"})

    issue = _issue_by_path(issues)["redis.key_prefix"]

    assert issue.value is None
    assert "key_prefix" in issue.reason
    assert issue.suggested_fix is not None
    assert "deployment-scoped" in issue.suggested_fix


def test_legacy_reservation_none_limiter_instance_id_reports_drain_step() -> None:
    reservation = {
        "usage": {"tokens": 1.0},
        "model_family": "gpt-4o",
        "limiter_instance_id": None,
    }

    issue = _issue_by_path(validate_config_for_v2_0(reservation))[
        "reservation.limiter_instance_id"
    ]

    assert issue.value is None
    assert "limiter_instance_id=None" in issue.reason
    assert issue.suggested_fix is not None
    assert "Drain" in issue.suggested_fix or "Drain".lower() in issue.suggested_fix


def test_capacity_reservation_object_with_legacy_owner_reports_drain_step() -> None:
    reservation = CapacityReservation(
        usage={"tokens": 1.0},
        model_family="gpt-4o",
        limiter_instance_id="limiter",
    )
    object.__setattr__(reservation, "limiter_instance_id", None)

    issue = _issue_by_path(validate_config_for_v2_0(reservation))[
        "reservation.limiter_instance_id"
    ]

    assert issue.reason == "legacy reservation has limiter_instance_id=None"
