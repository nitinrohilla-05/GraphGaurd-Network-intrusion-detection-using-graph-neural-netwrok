"""Unit tests for containment baselines and flow-level metrics (Owner Decisions 32 & 34)."""

from graphguard.core.models import (
    Detection,
    EdgeKey,
    GraphWindow,
    Incident,
)
from graphguard.detect.egraphsage import Detector
from graphguard.eval.baselines import (
    BlockAllFlaggedBaseline,
    IsolateHostBaseline,
    NoResponseBaseline,
)


class SimpleDetector(Detector):
    def __init__(self, flagged_edges):
        self.flagged_edges = flagged_edges

    def predict(self, window: GraphWindow) -> Detection:
        scores = {e.id: (0.90 if e.id in self.flagged_edges else 0.05) for e in window.edges}
        return Detection(
            edge_scores=scores,
            node_scores={},
            uncertainty={},
            flagged_edges=[e.id for e in window.edges if e.id in self.flagged_edges],
            flagged_nodes=[],
            edge_index_map={e.id: i for i, e in enumerate(window.edges)},
        )


def test_no_response_baseline() -> None:
    """NoResponseBaseline cuts no edges."""
    baseline = NoResponseBaseline()
    e0 = EdgeKey(src_ip="10.0.0.1", dst_ip="10.0.0.2", dst_port=80, proto="tcp")
    win = GraphWindow(
        window_id="w0",
        node_ids=["10.0.0.1", "10.0.0.2"],
        edge_index=[[0, 1]],
        edge_attr=[[1.0]],
        edges=[e0],
    )
    inc = Incident(incident_id="inc0", target_edges=[e0])
    det = SimpleDetector([e0.id])

    cut = baseline.solve(inc, win, det, lambda e, w: 0.1)
    assert len(cut.edges) == 0
    assert cut.total_cost == 0.0


def test_isolate_host_baseline_isolates_source_node() -> None:
    """Owner Correction 6: IsolateHost isolates the SOURCE host of flagged edges."""
    e_attack = EdgeKey(src_ip="10.0.0.1", dst_ip="10.0.0.2", dst_port=80, proto="tcp")
    e_benign_src = EdgeKey(src_ip="10.0.0.1", dst_ip="10.0.0.3", dst_port=53, proto="udp")
    e_unrelated = EdgeKey(src_ip="10.0.0.4", dst_ip="10.0.0.5", dst_port=443, proto="tcp")

    edges = [e_attack, e_benign_src, e_unrelated]
    node_ids = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4", "10.0.0.5"]
    node_map = {n: i for i, n in enumerate(node_ids)}

    win = GraphWindow(
        window_id="w_iso",
        node_ids=node_ids,
        edge_index=[[node_map[e.src_ip], node_map[e.dst_ip]] for e in edges],
        edge_attr=[[1.0]] * 3,
        edges=edges,
    )

    inc = Incident(incident_id="inc_iso", target_edges=[e_attack])
    det = SimpleDetector([e_attack.id])

    baseline = IsolateHostBaseline()
    cut = baseline.solve(inc, win, det, lambda e, w: 0.5)

    cut_ids = [ce.edge.id for ce in cut.edges]
    # Source node 10.0.0.1 incident edges should both be cut
    assert e_attack.id in cut_ids
    assert e_benign_src.id in cut_ids
    # Unrelated edge should NOT be cut
    assert e_unrelated.id not in cut_ids


def test_block_all_flagged_baseline() -> None:
    """BlockAllFlagged cuts exactly the flagged edges."""
    e_attack = EdgeKey(src_ip="10.0.0.1", dst_ip="10.0.0.2", dst_port=80, proto="tcp")
    e_benign = EdgeKey(src_ip="10.0.0.1", dst_ip="10.0.0.3", dst_port=53, proto="udp")

    edges = [e_attack, e_benign]
    node_ids = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    node_map = {n: i for i, n in enumerate(node_ids)}

    win = GraphWindow(
        window_id="w_blk",
        node_ids=node_ids,
        edge_index=[[node_map[e.src_ip], node_map[e.dst_ip]] for e in edges],
        edge_attr=[[1.0]] * 2,
        edges=edges,
    )

    inc = Incident(incident_id="inc_blk", target_edges=[e_attack])
    det = SimpleDetector([e_attack.id])

    baseline = BlockAllFlaggedBaseline()
    cut = baseline.solve(inc, win, det, lambda e, w: 0.5)

    assert len(cut.edges) == 1
    assert cut.edges[0].edge.id == e_attack.id
