"""Unit tests for DryRunEnforcer implementation and TTL expiry."""

from datetime import datetime, timedelta, timezone

from graphguard.core.models import Action, ActionStatus, ActionType, EdgeKey
from graphguard.enforce.dryrun import DryRunEnforcer


def _make_action(action_id: str, port: int = 80, expires_in_seconds: int = 3600) -> Action:
    now = datetime.now(timezone.utc)
    edge = EdgeKey(src_ip="192.168.1.100", dst_ip="10.0.0.10", dst_port=port, proto="tcp")
    return Action(
        action_id=action_id,
        type=ActionType.DROP,
        match=edge,
        status=ActionStatus.PENDING,
        created_at=now,
        expires_at=now + timedelta(seconds=expires_in_seconds),
    )


def test_dryrun_apply_and_idempotency() -> None:
    """Applying actions returns rule IDs and is idempotent."""
    enforcer = DryRunEnforcer()
    a1 = _make_action("act_1", port=80)
    a2 = _make_action("act_2", port=443)

    rule_ids = enforcer.apply([a1, a2])
    assert len(rule_ids) == 2
    assert "dryrun_rule_act_1" in rule_ids
    assert "dryrun_rule_act_2" in rule_ids
    assert a1.status == ActionStatus.APPLIED
    assert a2.status == ActionStatus.APPLIED
    assert enforcer.is_active("act_1")
    assert enforcer.is_active("act_2")

    # Apply again: idempotent, returns same rule IDs
    rule_ids_again = enforcer.apply([a1])
    assert rule_ids_again == ["dryrun_rule_act_1"]
    assert len(enforcer.active_rules) == 2


def test_dryrun_revert() -> None:
    """Reverting an action removes the active rule."""
    enforcer = DryRunEnforcer()
    a1 = _make_action("act_revert", port=80)
    enforcer.apply([a1])
    assert enforcer.is_active("act_revert")

    enforcer.revert("act_revert")
    assert not enforcer.is_active("act_revert")
    assert a1.status == ActionStatus.REVERTED

    # Reverting again is a no-op
    enforcer.revert("act_revert")
    assert not enforcer.is_active("act_revert")


def test_dryrun_reconcile_syncs_rules() -> None:
    """Reconcile aligns active rules to match the desired state."""
    enforcer = DryRunEnforcer()
    a1 = _make_action("act_keep", port=80)
    a2 = _make_action("act_remove", port=443)
    a3 = _make_action("act_add", port=22)

    # Initial state: a1 and a2 are active
    enforcer.apply([a1, a2])
    assert enforcer.is_active("act_keep")
    assert enforcer.is_active("act_remove")

    # Desired state: a1 and a3 (a2 should be removed, a3 should be added)
    active_rule_ids = enforcer.reconcile(desired=[a1, a3])

    assert "dryrun_rule_act_keep" in active_rule_ids
    assert "dryrun_rule_act_add" in active_rule_ids
    assert "dryrun_rule_act_remove" not in active_rule_ids
    assert enforcer.is_active("act_keep")
    assert enforcer.is_active("act_add")
    assert not enforcer.is_active("act_remove")


def test_dryrun_reconcile_purges_expired_actions() -> None:
    """Actions past expires_at (TTL) are purged during reconcile and marked EXPIRED."""
    enforcer = DryRunEnforcer()
    base_time = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)

    # Action that expires after 10 seconds
    edge = EdgeKey(src_ip="10.0.0.5", dst_ip="10.0.0.1", dst_port=80, proto="tcp")
    a_expiring = Action(
        action_id="act_ttl",
        type=ActionType.RATE_LIMIT,
        match=edge,
        status=ActionStatus.PENDING,
        created_at=base_time,
        expires_at=base_time + timedelta(seconds=10),
        rate_limit_kbps=64,
    )
    # Action that expires after 1 hour
    a_persisting = Action(
        action_id="act_persist",
        type=ActionType.DROP,
        match=EdgeKey(src_ip="10.0.0.6", dst_ip="10.0.0.1", dst_port=80, proto="tcp"),
        status=ActionStatus.PENDING,
        created_at=base_time,
        expires_at=base_time + timedelta(seconds=3600),
    )

    # Apply both at base_time
    enforcer.apply([a_expiring, a_persisting], now=base_time)
    assert enforcer.is_active("act_ttl")
    assert enforcer.is_active("act_persist")

    # Advance time by 30 seconds
    future_time = base_time + timedelta(seconds=30)

    # Reconcile at future_time with both desired
    active_rules = enforcer.reconcile(desired=[a_expiring, a_persisting], now=future_time)

    assert not enforcer.is_active("act_ttl")
    assert a_expiring.status == ActionStatus.EXPIRED
    assert enforcer.is_active("act_persist")
    assert "dryrun_rule_act_persist" in active_rules
    assert "dryrun_rule_act_ttl" not in active_rules
