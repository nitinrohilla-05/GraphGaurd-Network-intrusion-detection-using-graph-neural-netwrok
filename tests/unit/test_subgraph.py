"""Unit tests for k-hop subgraph extraction and candidate edge restriction."""

from graphguard.containment.subgraph import extract_khop_subgraph
from graphguard.core.models import Detection, EdgeKey, GraphWindow


def test_extract_khop_subgraph_and_candidate_restriction() -> None:
    """Validate 2-hop induced subgraph extraction and Decision 29 candidate restrictions."""
    # Graph layout:
    # e0: 10.0.0.1 -> 10.0.0.2 (Attack, score 0.90) [Flagged]
    # e1: 10.0.0.1 -> 10.0.0.3 (Benign, score 0.10) [Incident to flagged source 10.0.0.1]
    # e2: 10.0.0.2 -> 10.0.0.4 (Benign, score 0.40) [score >= tau_low 0.30 -> Candidate]
    # e3: 10.0.0.4 -> 10.0.0.5 (Benign, score 0.05) [score < 0.30, not incident -> NOT Candidate]
    # e4: 10.0.0.8 -> 10.0.0.9 (Distant benign, > 2 hops -> Excluded from subgraph)

    e0 = EdgeKey(src_ip="10.0.0.1", dst_ip="10.0.0.2", dst_port=80, proto="tcp")
    e1 = EdgeKey(src_ip="10.0.0.1", dst_ip="10.0.0.3", dst_port=53, proto="udp")
    e2 = EdgeKey(src_ip="10.0.0.2", dst_ip="10.0.0.4", dst_port=443, proto="tcp")
    e3 = EdgeKey(src_ip="10.0.0.4", dst_ip="10.0.0.5", dst_port=80, proto="tcp")
    e4 = EdgeKey(src_ip="10.0.0.8", dst_ip="10.0.0.9", dst_port=80, proto="tcp")

    edges = [e0, e1, e2, e3, e4]
    node_ids = [
        "10.0.0.1",
        "10.0.0.2",
        "10.0.0.3",
        "10.0.0.4",
        "10.0.0.5",
        "10.0.0.8",
        "10.0.0.9",
    ]
    node_map = {ip: i for i, ip in enumerate(node_ids)}
    edge_index = [[node_map[e.src_ip], node_map[e.dst_ip]] for e in edges]
    edge_attr = [[1.0] * 4 for _ in edges]

    window = GraphWindow(
        window_id="win_test",
        node_ids=node_ids,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edges=edges,
    )

    det = Detection(
        edge_scores={
            e0.id: 0.90,
            e1.id: 0.10,
            e2.id: 0.40,
            e3.id: 0.05,
            e4.id: 0.01,
        },
        node_scores={"10.0.0.1": 0.90},
        uncertainty={},
        flagged_edges=[e0.id],
        flagged_nodes=["10.0.0.1"],
        edge_index_map={e.id: i for i, e in enumerate(edges)},
    )

    region = extract_khop_subgraph(
        window=window,
        flagged_edge_ids=[e0.id],
        detection=det,
        k_hops=2,
        tau_low=0.30,
    )

    # 1. Distant edge e4 should be excluded from 2-hop subgraph
    sub_edge_ids = [e.id for e in region.window.edges]
    assert e4.id not in sub_edge_ids
    assert e0.id in sub_edge_ids
    assert e1.id in sub_edge_ids
    assert e2.id in sub_edge_ids

    # 2. Candidate restrictions check (Decision 29):
    # Candidate edges within the subgraph:
    cand_edges = [region.window.edges[idx].id for idx in region.candidate_edge_indices]

    # e0 is candidate (score 0.90 >= 0.30 and flagged)
    assert e0.id in cand_edges
    # e1 is candidate (incident to source node 10.0.0.1 of flagged edge)
    assert e1.id in cand_edges
    # e2 is candidate (score 0.40 >= 0.30)
    assert e2.id in cand_edges
    # e3 is NEVER a candidate: score 0.05 < 0.30 and neither endpoint is flagged src 10.0.0.1
    assert e3.id not in cand_edges
