"""k-hop induced subgraph extraction around flagged threat targets (Decision 29 & 30)."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from graphguard.core.models import Detection, EdgeKey, GraphWindow


@dataclass
class SubgraphRegion:
    """Induced k-hop subgraph containing flagged targets and candidate edges."""

    window: GraphWindow
    original_edge_indices: List[int]
    node_map: Dict[str, int]
    flagged_edge_ids: Set[str]
    flagged_src_nodes: Set[str]
    candidate_edge_indices: List[int]


def extract_khop_subgraph(
    window: GraphWindow,
    flagged_edge_ids: List[str],
    detection: Optional[Detection] = None,
    k_hops: int = 2,
    tau_low: float = 0.30,
) -> SubgraphRegion:
    """Extract induced k-hop subgraph around flagged threat edges.

    Owner Decision 29:
    Candidate edges that may be cut are strictly restricted to:
    (a) attack score >= tau_low, OR
    (b) incident to a source node of a flagged edge.
    Edges between two unflagged nodes with score < tau_low are never candidates.
    """
    if not window.edges or not flagged_edge_ids:
        # Return empty region or minimal representation
        return SubgraphRegion(
            window=GraphWindow(
                window_id=f"{window.window_id}_sub",
                start_time=window.start_time,
                end_time=window.end_time,
                node_ids=[],
                edge_index=[],
                edge_attr=[],
                edges=[],
                labels=None,
            ),
            original_edge_indices=[],
            node_map={},
            flagged_edge_ids=set(),
            flagged_src_nodes=set(),
            candidate_edge_indices=[],
        )

    flagged_set = set(flagged_edge_ids)
    flagged_src_nodes: Set[str] = set()

    # Identify seed nodes from flagged edges
    seed_nodes: Set[str] = set()
    for e in window.edges:
        if e.id in flagged_set:
            seed_nodes.add(e.src_ip)
            seed_nodes.add(e.dst_ip)
            flagged_src_nodes.add(e.src_ip)

    # Grow k hops
    current_nodes = set(seed_nodes)
    for _ in range(k_hops):
        next_nodes = set(current_nodes)
        for e in window.edges:
            if e.src_ip in current_nodes or e.dst_ip in current_nodes:
                next_nodes.add(e.src_ip)
                next_nodes.add(e.dst_ip)
        current_nodes = next_nodes

    # Extract all induced edges whose endpoints both fall within current_nodes
    sub_edges: List[EdgeKey] = []
    sub_edge_attr: List[List[float]] = []
    sub_labels: Optional[List[int]] = [] if window.labels is not None else None
    original_edge_indices: List[int] = []

    for orig_idx, e in enumerate(window.edges):
        if e.src_ip in current_nodes and e.dst_ip in current_nodes:
            sub_edges.append(e)
            sub_edge_attr.append(window.edge_attr[orig_idx])
            if sub_labels is not None and window.labels is not None:
                sub_labels.append(window.labels[orig_idx])
            original_edge_indices.append(orig_idx)

    # Build local node indexing
    unique_nodes = sorted(list(current_nodes))
    node_map = {nid: i for i, nid in enumerate(unique_nodes)}

    sub_edge_index: List[List[int]] = [[node_map[e.src_ip], node_map[e.dst_ip]] for e in sub_edges]

    sub_window = GraphWindow(
        window_id=f"{window.window_id or 'win'}_sub",
        t_start=window.t_start,
        t_end=window.t_end,
        node_ids=unique_nodes,
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        edges=sub_edges,
        labels=sub_labels,
    )

    # Determine candidate edge indices per Decision 29
    candidate_edge_indices: List[int] = []
    for local_idx, e in enumerate(sub_edges):
        score = 0.0
        if detection and e.id in detection.edge_scores:
            score = detection.edge_scores[e.id]
        elif e.id in flagged_set:
            score = 1.0

        is_incident_to_flagged_src = e.src_ip in flagged_src_nodes or e.dst_ip in flagged_src_nodes

        if score >= tau_low or is_incident_to_flagged_src:
            candidate_edge_indices.append(local_idx)

    return SubgraphRegion(
        window=sub_window,
        original_edge_indices=original_edge_indices,
        node_map=node_map,
        flagged_edge_ids=flagged_set,
        flagged_src_nodes=flagged_src_nodes,
        candidate_edge_indices=candidate_edge_indices,
    )
