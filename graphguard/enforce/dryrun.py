"""In-memory safe dry-run enforcer implementation."""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from graphguard.core.interfaces import Enforcer
from graphguard.core.models import Action, ActionStatus

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_expired(action: Action, current_time: datetime) -> bool:
    """Return True if action has expired relative to current_time."""
    if action.expires_at is None:
        return False
    exp = (
        action.expires_at
        if action.expires_at.tzinfo
        else action.expires_at.replace(tzinfo=timezone.utc)
    )
    return exp <= current_time


class DryRunEnforcer(Enforcer):
    """Safe, in-memory reference Enforcer implementation.

    Simulates network dataplane rule installation without modifying real firewall
    or SDN switch state. Handles TTL expiration and reconciliation.
    """

    def __init__(self) -> None:
        # active_rules: rule_id -> Action
        self._active_rules: Dict[str, Action] = {}
        # action_id -> rule_id
        self._action_to_rule: Dict[str, str] = {}

    @property
    def active_rules(self) -> Dict[str, Action]:
        """Dictionary of currently installed active rules."""
        return dict(self._active_rules)

    def is_active(self, action_id: str) -> bool:
        """Check if an action is currently installed and active."""
        return action_id in self._action_to_rule

    def apply(self, actions: List[Action], now: Optional[datetime] = None) -> List[str]:
        """Apply actions idempotently; return list of installed rule IDs."""
        current_time = now or _utc_now()
        rule_ids: List[str] = []

        for action in actions:
            if _is_expired(action, current_time):
                action.status = ActionStatus.EXPIRED
                logger.info(f"Skipping expired action {action.action_id}")
                continue

            rule_id = f"dryrun_rule_{action.action_id}"
            if action.action_id in self._action_to_rule:
                # Already installed, idempotent
                rule_ids.append(rule_id)
                continue

            action.status = ActionStatus.APPLIED
            self._active_rules[rule_id] = action
            self._action_to_rule[action.action_id] = rule_id
            rule_ids.append(rule_id)
            logger.info(f"[DryRun] Applied {rule_id} for {action.action_id} ({action.type.value})")

        return rule_ids

    def revert(self, action_id: str) -> None:
        """Revert a previously installed action rule."""
        if action_id in self._action_to_rule:
            rule_id = self._action_to_rule.pop(action_id)
            if rule_id in self._active_rules:
                action = self._active_rules.pop(rule_id)
                action.status = ActionStatus.REVERTED
                logger.info(f"[DryRun] Reverted rule {rule_id} for action {action_id}")

    def reconcile(self, desired: List[Action], now: Optional[datetime] = None) -> List[str]:
        """Align active rules with desired state, purging expired and stale rules."""
        current_time = now or _utc_now()

        # Step 1: Detect and purge expired rules from active set
        expired_action_ids: List[str] = []
        for rule_id, action in list(self._active_rules.items()):
            if _is_expired(action, current_time):
                action.status = ActionStatus.EXPIRED
                expired_action_ids.append(action.action_id)

        for act_id in expired_action_ids:
            logger.info(f"[DryRun] Reconciler removing expired action {act_id}")
            self.revert(act_id)

        # Step 2: Separate valid desired actions from expired desired actions
        valid_desired_map: Dict[str, Action] = {}
        for action in desired:
            if _is_expired(action, current_time):
                action.status = ActionStatus.EXPIRED
                continue
            valid_desired_map[action.action_id] = action

        # Step 3: Remove active rules that are not in valid desired set
        currently_active_action_ids = list(self._action_to_rule.keys())
        for act_id in currently_active_action_ids:
            if act_id not in valid_desired_map:
                logger.info(f"[DryRun] Reconciler removing stale action {act_id}")
                self.revert(act_id)

        # Step 4: Apply missing desired actions
        to_apply: List[Action] = []
        for act_id, action in valid_desired_map.items():
            if act_id not in self._action_to_rule:
                to_apply.append(action)

        if to_apply:
            self.apply(to_apply, now=current_time)

        return list(self._active_rules.keys())
