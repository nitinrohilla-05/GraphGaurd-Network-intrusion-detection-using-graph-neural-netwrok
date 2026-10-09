#!/usr/bin/env python3
"""GraphGuard End-to-End Demo Script.

Demonstrates GNN Network Intrusion Detection (E-GraphSAGE) and
Cost-Aware Counterfactual Minimal Cut Containment (CVC loop) on
authentic NetFlow sample traffic.
"""

from pathlib import Path
import sys
import pandas as pd

from graphguard.containment.cost import EdgeCostEvaluator
from graphguard.containment.cut_solver import CounterfactualCutSolver
from graphguard.core.models import (
    Incident,
    IncidentSeverity,
    IncidentStatus,
)
from graphguard.detect.registry import ModelRegistry
from graphguard.graph.window_builder import WindowBuilder


def main() -> None:
    print("=" * 72)
    print("  GraphGuard: GNN Intrusion Detection & Counterfactual Containment")
    print("=" * 72)

    repo_root = Path(__file__).resolve().parent.parent
    sample_csv = repo_root / "data" / "sample_nf_v2.csv"
    models_dir = repo_root / "models"

    if not sample_csv.exists():
        print(f"[!] Error: Sample dataset not found at {sample_csv}")
        sys.exit(1)

    # 1. Load sample data
    print(f"\n[1/5] Ingesting authentic NetFlow flows from: data/{sample_csv.name}")
    df = pd.read_csv(sample_csv, nrows=300)
    print(f"      Ingested {len(df):,} flows ({len(df.columns)} NetFlow features)")

    # 2. Load pre-trained E-GraphSAGE model from registry
    print("\n[2/5] Loading pre-trained E-GraphSAGE model from registry...")
    registry = ModelRegistry(str(models_dir))
    try:
        detector, feature_extractor, metadata = registry.load_model("egraphsage", version="latest")
        print(f"      Model version            : {metadata.get('version', 'latest')}")
        print(f"      Calibrated threshold (tau): {detector.tau:.3f}")
        print(f"      Supported classes        : {len(detector.classes)} categories")
    except Exception as e:
        print(f"[!] Failed to load model from registry: {e}")
        sys.exit(1)

    # 3. Build Graph Window
    print("\n[3/5] Transforming NetFlow flows into Communication Graph Topology...")
    builder = WindowBuilder(chunk_size=len(df), window_duration_seconds=None)
    windows = builder.build_windows(df, feature_extractor)
    window, labels, attack_names = windows[0]
    print(f"      Nodes (IP Endpoints)     : {len(window.node_ids)}")
    print(f"      Edges (Network Flows)    : {len(window.edges)}")

    # 4. Run GNN Edge Classification
    print("\n[4/5] Running GNN Edge Classification (E-GraphSAGE)...")
    detection = detector.predict(window)
    flagged_edges = [
        edge for edge in window.edges
        if detection.edge_scores.get(edge.id, 0.0) >= detector.tau
    ]
    print(f"      Flagged intrusions       : {len(flagged_edges)} / {len(window.edges)} flows")
    if flagged_edges:
        print("      Top detected threats:")
        for edge in flagged_edges[:5]:
            score = detection.edge_scores[edge.id]
            print(f"       -> {edge.src_ip} -> {edge.dst_ip}:{edge.dst_port} ({edge.proto.upper()}) | Malicious Score: {score:.3f}")

    # 5. Counterfactual Minimal Cut Containment (CVC loop)
    print("\n[5/5] Computing Cost-Aware Counterfactual Minimal Cut (CVC Loop: N1 + N2)...")
    incident_edges = flagged_edges if flagged_edges else window.edges[:2]
    incident = Incident(
        incident_id="DEMO-INCIDENT-001",
        severity=IncidentSeverity.HIGH if flagged_edges else IncidentSeverity.LOW,
        status=IncidentStatus.OPEN,
        target_edges=incident_edges,
    )

    cost_evaluator = EdgeCostEvaluator(alpha=0.6, beta=0.4)
    solver = CounterfactualCutSolver(
        tau_low=detector.tau * 0.75,
        tau=detector.tau,
        lambda1=1.0,
        lambda2=0.5,
        max_steps=50,
    )

    cut_set = solver.solve(
        incident=incident,
        window=window,
        detector=detector,
        cost_fn=cost_evaluator.compute_cost,
    )

    print("\n" + "-" * 72)
    print("  CONTAINMENT DECISION SUMMARY")
    print("-" * 72)
    print(f"  Incident ID              : {incident.incident_id}")
    print(f"  Severity                 : {incident.severity.value.upper()}")
    print(f"  Total Cut Edges          : {len(cut_set.edges)}")
    print(f"  Containment Probability  : {cut_set.containment_probability:.1%}")
    print(f"  Disruption Cost Penalty  : {cut_set.total_cost:.4f}")
    print("\n  Prescribed Interventions:")
    for cut_edge in cut_set.edges[:5]:
        e = cut_edge.edge
        print(f"   [{cut_edge.action_type.value}] Flow {e.src_ip} -> {e.dst_ip}:{e.dst_port} ({e.proto.upper()}) | Cost: {cut_edge.cost:.2f} | Mask: {cut_edge.mask_weight:.3f}")

    print("\n" + "=" * 72)
    print("  SUCCESS: GraphGuard pipeline executed cleanly end-to-end!")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
