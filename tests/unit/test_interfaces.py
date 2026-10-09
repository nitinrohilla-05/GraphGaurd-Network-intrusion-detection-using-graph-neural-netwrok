"""Unit tests verifying all 8 core component interfaces (ABCs)."""

import inspect
from abc import ABC
from typing import List

import pytest

from graphguard.core.interfaces import (
    ActionPolicy,
    CutSolver,
    Detector,
    Enforcer,
    LinkPredictor,
    ReactionAnalyzer,
    RollbackManager,
    Verifier,
)
from graphguard.core.models import (
    Action,
)

CORE_ABCS = [
    Detector,
    CutSolver,
    ActionPolicy,
    Enforcer,
    Verifier,
    ReactionAnalyzer,
    RollbackManager,
    LinkPredictor,
]


def test_core_abcs_count_and_types() -> None:
    """Ensure exactly 8 core ABC interfaces are exported and inherit from abc.ABC."""
    assert len(CORE_ABCS) == 8, f"Expected exactly 8 ABCs, found {len(CORE_ABCS)}"
    for cls in CORE_ABCS:
        assert inspect.isabstract(cls), f"{cls.__name__} should be an abstract class"
        assert issubclass(cls, ABC), f"{cls.__name__} must subclass abc.ABC"


def test_core_abcs_cannot_be_instantiated_directly() -> None:
    """Attempting to instantiate any of the 8 ABCs directly must raise TypeError."""
    for cls in CORE_ABCS:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            cls()  # type: ignore[abstract]


def test_enforcer_contract_enforcement() -> None:
    """Subclassing Enforcer without implementing all abstract methods raises TypeError."""

    class IncompleteEnforcer(Enforcer):
        def apply(self, actions: List[Action]) -> List[str]:
            return []

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteEnforcer()  # type: ignore[abstract]

    class CompleteEnforcer(Enforcer):
        def apply(self, actions: List[Action]) -> List[str]:
            return ["rule_1"]

        def revert(self, action_id: str) -> None:
            pass

        def reconcile(self, desired: List[Action]) -> List[str]:
            return ["rule_1"]

    instance = CompleteEnforcer()
    assert instance.apply([]) == ["rule_1"]
    assert instance.reconcile([]) == ["rule_1"]
