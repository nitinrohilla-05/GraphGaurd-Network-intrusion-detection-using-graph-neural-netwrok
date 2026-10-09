"""Unit tests for CounterfactualCutSolver (N1) and GreedyCutSolver (Decisions 29, 30, 31, 36)."""

import torch
import torch.nn as nn

from graphguard.containment.cut_solver import (
    CounterfactualCutSolver,
    GreedyCutSolver,
)
from graphguard.core.interfaces import Detector
from graphguard.core.models import (
    Detection,
    EdgeKey,
    GraphWindow,
    Incident,
    IncidentSeverity,
    IncidentStatus,
)
from graphguard.detect.egraphsage import EdgeSAGEConv


def test_mask_weighted_aggregation_equivalence_to_deletion() -> None:
    """Owner Correction 3: Unit-test that M_e = 0 gives the same output as deleting it."""
    torch.manual_seed(42)

    in_dim = 4
    edge_dim = 6
    out_dim = 8

    conv = EdgeSAGEConv(in_dim, edge_dim, out_dim)
    conv.eval()

    num_nodes = 4
    x = torch.randn(num_nodes, in_dim)

    # 3 edges:
    # 0 -> 1, 1 -> 2, 2 -> 3
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, edge_dim)

    # Case A: Mask edge index 2 with M=0 (edges 0 and 1 have M=1, edge 2 has M=0)
    edge_mask = torch.tensor([1.0, 1.0, 0.0])
    with torch.no_grad():
        out_masked = conv(x, edge_index, edge_attr, edge_mask=edge_mask)

    # Case B: Physically delete edge 2 (only edges 0 and 1 present in edge_index)
    edge_index_deleted = edge_index[:, :2]
    edge_attr_deleted = edge_attr[:2]
    edge_mask_deleted = torch.tensor([1.0, 1.0])
    with torch.no_grad():
        out_deleted = conv(x, edge_index_deleted, edge_attr_deleted, edge_mask=edge_mask_deleted)

    # Validate mathematical equivalence: max diff must be negligible (< 1e-6)
    diff = torch.max(torch.abs(out_masked - out_deleted)).item()
    assert diff < 1e-6, f"Mask M=0 did not match physical deletion! Max diff was {diff}"


class MockControlledDetector(Detector):
    """Mock detector for hand-built test graph with known ground truth (Decision 36)."""

    def __init__(self, attack_edge_id: str, tau: float = 0.50) -> None:
        self.attack_edge_id = attack_edge_id
        self.tau = tau
        self.benign_class_idx = 0

        # Small mock neural net where edge representation depends on edge features
        # and msg passing, with gradient toward attack edge
        class MockNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(2, 2)

            def forward(
                self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_mask: torch.Tensor | None = None,
            ) -> torch.Tensor:
                is_atk = edge_attr[:, 0:1]  # [E, 1]
                if edge_mask is not None:
                    m = edge_mask.view(-1, 1)
                else:
                    m = torch.ones_like(is_atk)
                atk_logit = 3.0 * is_atk * m - 3.0 * (1.0 - is_atk)
                benign_logit = -atk_logit
                return torch.cat([benign_logit, atk_logit], dim=-1)

        self.model = MockNet()

    def predict(self, window: GraphWindow) -> Detection:
        edge_scores = {}
        flagged = []
        for i, edge in enumerate(window.edges):
            # Attack edge scores 0.95 unless cut
            if edge.id == self.attack_edge_id:
                score = 0.95
            else:
                score = 0.05
            edge_scores[edge.id] = score
            if score >= self.tau:
                flagged.append(edge.id)

        return Detection(
            edge_scores=edge_scores,
            node_scores={},
            uncertainty={},
            flagged_edges=flagged,
            flagged_nodes=[],
            edge_index_map={e.id: i for i, e in enumerate(window.edges)},
        )


def test_n1_does_not_cut_benign_neighbor_edges() -> None:
    """Owner Correction 8: Verify N1 cuts attack edge and does NOT cut benign neighbor edges."""
    # Hand-built graph:
    # e_atk: 10.0.0.1 -> 10.0.0.2 (Attacker -> Victim, score 0.95, cost 0.20)
    # e_benign_srv: 10.0.0.2 -> 10.0.0.50 (Victim -> DB, score 0.05, cost 0.90)
    # e_benign_dns: 10.0.0.1 -> 10.0.0.3 (Attacker -> DNS, score 0.05, cost 0.80)

    e_atk = EdgeKey(src_ip="10.0.0.1", dst_ip="10.0.0.2", dst_port=80, proto="tcp")
    e_benign_srv = EdgeKey(src_ip="10.0.0.2", dst_ip="10.0.0.50", dst_port=3306, proto="tcp")
    e_benign_dns = EdgeKey(src_ip="10.0.0.1", dst_ip="10.0.0.3", dst_port=53, proto="udp")

    edges = [e_atk, e_benign_srv, e_benign_dns]
    node_ids = ["10.0.0.1", "10.0.0.2", "10.0.0.50", "10.0.0.3"]
    node_map = {nid: i for i, nid in enumerate(node_ids)}
    edge_index = [[node_map[e.src_ip], node_map[e.dst_ip]] for e in edges]
    # Column 0 indicates ground truth attack (1.0 vs 0.0)
    edge_attr = [[1.0, 0.5], [0.0, 0.8], [0.0, 0.2]]

    window = GraphWindow(
        window_id="win_handbuilt",
        node_ids=node_ids,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edges=edges,
    )

    detector = MockControlledDetector(attack_edge_id=e_atk.id, tau=0.50)

    def cost_fn(edge: EdgeKey, w: GraphWindow) -> float:
        if edge.id == e_atk.id:
            return 0.20
        elif edge.id == e_benign_srv.id:
            return 0.90  # Critical DB link
        else:
            return 0.80  # Critical DNS link

    incident = Incident(
        incident_id="inc_mock",
        severity=IncidentSeverity.HIGH,
        status=IncidentStatus.OPEN,
        target_edges=[e_atk],
    )

    solver = CounterfactualCutSolver(
        tau_low=0.30,
        tau=0.50,
        lambda1=1.0,
        lambda2=0.5,
        max_steps=100,
    )

    cut_set = solver.solve(incident, window, detector, cost_fn)

    cut_ids = [ce.edge.id for ce in cut_set.edges]

    # Verify N1 cuts the attack edge
    assert e_atk.id in cut_ids
    # Verify N1 does NOT cut the benign neighbor edges!
    assert e_benign_srv.id not in cut_ids
    assert e_benign_dns.id not in cut_ids

    # Containment must be successful
    assert cut_set.containment_probability == 1.0


def test_hard_removal_success_condition_and_remediation() -> None:
    """Owner Correction 2: Verify hard-removal verification and remediation."""
    # Graph with 2 attack edges
    e1 = EdgeKey(src_ip="10.0.0.1", dst_ip="10.0.0.2", dst_port=80, proto="tcp")
    e2 = EdgeKey(src_ip="10.0.0.1", dst_ip="10.0.0.3", dst_port=80, proto="tcp")

    edges = [e1, e2]
    node_ids = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    node_map = {nid: i for i, nid in enumerate(node_ids)}
    edge_index = [[node_map[e.src_ip], node_map[e.dst_ip]] for e in edges]
    edge_attr = [[1.0, 0.5], [1.0, 0.5]]

    window = GraphWindow(
        window_id="win_multi_attack",
        node_ids=node_ids,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edges=edges,
    )

    class MultiAttackDetector(Detector):
        def predict(self, w: GraphWindow) -> Detection:
            scores = {e.id: 0.90 for e in w.edges}
            return Detection(
                edge_scores=scores,
                node_scores={},
                uncertainty={},
                flagged_edges=[e.id for e in w.edges],
                flagged_nodes=[],
                edge_index_map={e.id: i for i, e in enumerate(w.edges)},
            )

    det = MultiAttackDetector()
    incident = Incident(
        incident_id="inc_multi",
        target_edges=[e1, e2],
    )

    solver = GreedyCutSolver(tau_low=0.30, tau=0.50)
    cut_set = solver.solve(incident, window, det, lambda e, w: 0.5)

    assert len(cut_set.edges) == 2
    assert cut_set.containment_probability == 1.0
