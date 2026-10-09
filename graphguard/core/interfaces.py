"""Abstract Base Classes (ABCs) defining the 8 core component contracts for GraphGuard."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, List, Optional, Tuple

from graphguard.core.models import (
    Action,
    CutSet,
    Detection,
    EdgeKey,
    GraphWindow,
    Incident,
    ReactionReport,
    Verdict,
)


class Detector(ABC):
    """GNN inference detector interface."""

    @abstractmethod
    def predict(self, window: GraphWindow) -> Detection:
        """Score graph window returning edge scores (by EdgeKey.id), node scores, uncertainty."""
        pass


class CutSolver(ABC):
    """Counterfactual minimal cut solver interface (N1)."""

    @abstractmethod
    def solve(
        self,
        window: GraphWindow,
        detection: Detection,
        cost_fn: Optional[Callable[[EdgeKey, GraphWindow], float]] = None,
        extra_candidates: Optional[List[EdgeKey]] = None,
    ) -> CutSet:
        """Compute the cost-aware counterfactual minimal cut set."""
        pass


class ActionPolicy(ABC):
    """Graded action decision policy interface (N3)."""

    @abstractmethod
    def decide(self, cut_set: CutSet, detection: Detection) -> List[Action]:
        """Map cut set edges to graded network actions (DROP, RATE_LIMIT, REDIRECT_TO_DECOY)."""
        pass


class Enforcer(ABC):
    """Network rule enforcement and reconciliation interface."""

    @abstractmethod
    def apply(self, actions: List[Action]) -> List[str]:
        """Apply actions idempotently to network dataplane; return installed rule IDs."""
        pass

    @abstractmethod
    def revert(self, action_id: str) -> None:
        """Revert a previously installed action rule."""
        pass

    @abstractmethod
    def reconcile(self, desired: List[Action]) -> List[str]:
        """Reconciliation loop: align active rules with desired state and purge expired."""
        pass


class Verifier(ABC):
    """Model-in-the-loop verification interface (N2)."""

    @abstractmethod
    def step(self, incident: Incident, window: GraphWindow) -> Verdict:
        """Verify containment state on the subsequent traffic window using detector re-scoring."""
        pass


class ReactionAnalyzer(ABC):
    """Intervention response and evasion analysis interface (N6)."""

    @abstractmethod
    def analyze(
        self,
        source: str,
        pre_windows: List[GraphWindow],
        post_windows: List[GraphWindow],
    ) -> ReactionReport:
        """Analyze source behavior before vs after action, outputting evasion score."""
        pass


class RollbackManager(ABC):
    """What-if rollback simulation and false alarm restoration interface (N4)."""

    @abstractmethod
    def tick(self, now: datetime) -> List[str]:
        """Evaluate what-if graphs for active actions, restoring benign false alarms."""
        pass


class LinkPredictor(ABC):
    """Temporal link forecasting interface for lateral movement pre-containment (N5)."""

    @abstractmethod
    def predict(self, window: GraphWindow, source: str, k: int) -> List[Tuple[str, int, float]]:
        """Forecast top-k candidate future edges: list of (dst_ip, dst_port, probability)."""
        pass
