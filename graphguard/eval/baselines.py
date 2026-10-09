"""Tabular baseline (RandomForest) and containment baselines (SPEC Sections 12 & 14)."""

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from graphguard.core.interfaces import CutSolver, Detector
from graphguard.core.models import CutEdge, CutSet, GraphWindow, Incident

logger = logging.getLogger(__name__)


class TabularRandomForestBaseline:
    """RandomForest classifier trained on flat edge feature vectors without graph topology."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        random_state: int = 42,
    ) -> None:
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.trained_samples_count: int = 0

    def fit(
        self,
        X_train: np.ndarray,
        y_train_binary: np.ndarray,
        max_samples: Optional[int] = 50000,
        random_state: int = 42,
    ) -> "TabularRandomForestBaseline":
        """Fit RandomForest with optional stratified subsampling for scalability (Decision 27)."""
        if max_samples is not None and len(X_train) > max_samples:
            logger.info(
                f"Subsampling training set for RandomForest from {len(X_train)} "
                f"to {max_samples} samples (stratified, seed={random_state})."
            )
            X_sub, _, y_sub, _ = train_test_split(
                X_train,
                y_train_binary,
                train_size=max_samples,
                stratify=y_train_binary,
                random_state=random_state,
            )
        else:
            X_sub, y_sub = X_train, y_train_binary

        self.trained_samples_count = len(X_sub)
        self.rf.fit(X_sub, y_sub)
        return self

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test_binary: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate baseline on test features."""
        preds = self.rf.predict(X_test)

        precision = float(precision_score(y_test_binary, preds, zero_division=0))
        recall = float(recall_score(y_test_binary, preds, zero_division=0))
        f1 = float(f1_score(y_test_binary, preds, zero_division=0))

        return {
            "baseline_model": "RandomForest",
            "trained_samples": self.trained_samples_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


class NoResponseBaseline(CutSolver):
    """Control baseline: performs zero defensive cuts."""

    def solve(
        self,
        incident: Incident,
        window: GraphWindow,
        detector: Detector,
        cost_fn: Any,
    ) -> CutSet:
        return CutSet(
            edges=[],
            total_cost=0.0,
            containment_probability=0.0,
            status_message="No response applied",
        )


class IsolateHostBaseline(CutSolver):
    """Isolates SOURCE host(s) of flagged edges by cutting all incident edges (Decision 34)."""

    def solve(
        self,
        incident: Incident,
        window: GraphWindow,
        detector: Detector,
        cost_fn: Any,
    ) -> CutSet:
        flagged_edge_ids = {e.id for e in incident.target_edges}
        if not flagged_edge_ids:
            det = detector.predict(window)
            flagged_edge_ids = set(det.flagged_edges)

        # Source host(s) of flagged edges
        flagged_src_hosts = set()
        for edge in window.edges:
            if edge.id in flagged_edge_ids:
                flagged_src_hosts.add(edge.src_ip)

        cut_edges = []
        total_cost = 0.0
        for edge in window.edges:
            if edge.src_ip in flagged_src_hosts or edge.dst_ip in flagged_src_hosts:
                cost = float(cost_fn(edge, window)) if callable(cost_fn) else 0.0
                total_cost += cost
                cut_edges.append(
                    CutEdge(
                        edge=edge,
                        mask_weight=0.0,
                        cost=round(cost, 4),
                        virtual=False,
                    )
                )

        # Post-cut remaining verification (Decision 44)
        cut_ids = {ce.edge.id for ce in cut_edges}
        kept = [e for e in window.edges if e.id not in cut_ids]
        eff_tau = getattr(detector, "tau", 0.50)
        drift_count = 0
        drift_labels = []
        if not kept:
            success = True
            max_rem = 0.0
            fail_reason = "None"
        else:
            nodes = sorted(list({e.src_ip for e in kept} | {e.dst_ip for e in kept}))
            nmap = {n: i for i, n in enumerate(nodes)}
            rem_win = GraphWindow(
                window_id=f"{window.window_id}_iso_check",
                start_time=window.start_time,
                end_time=window.end_time,
                node_ids=nodes,
                edge_index=[[nmap[e.src_ip] for e in kept], [nmap[e.dst_ip] for e in kept]],
                edge_attr=[
                    window.edge_attr[i] for i, e in enumerate(window.edges) if e.id not in cut_ids
                ],
                edges=kept,
            )
            rem_det = detector.predict(rem_win)
            cand_rem_scores = [
                float(rem_det.edge_scores.get(e.id, 0.0)) for e in kept if e.id in flagged_edge_ids
            ]
            max_rem = max(cand_rem_scores) if cand_rem_scores else 0.0
            # Containment: every remaining edge that was a candidate before now scores < tau
            success = max_rem < eff_tau
            fail_reason = (
                "None"
                if success
                else f"Remaining candidate score {max_rem:.4f} >= tau {eff_tau:.4f}"
            )
            # Score drift: edges not flagged before that rose >= tau
            for i, e in enumerate(window.edges):
                if e.id not in cut_ids and e.id not in flagged_edge_ids:
                    sc = float(rem_det.edge_scores.get(e.id, 0.0))
                    if sc >= eff_tau:
                        drift_count += 1
                        if window.labels is not None:
                            drift_labels.append(window.labels[i])

        msg = (
            f"Isolated {len(flagged_src_hosts)} source host(s) with "
            f"{len(cut_edges)} edge cuts (success={success}, drift={drift_count})"
        )
        return CutSet(
            edges=cut_edges,
            total_cost=round(total_cost, 4),
            containment_probability=1.0 if success else 0.0,
            status_message=msg,
            metadata={
                "candidate_count": len(flagged_edge_ids),
                "initial_cut_size": 0,
                "final_cut_size": len(cut_edges),
                "iterations_used": 0,
                "max_remaining_unblocked_score": round(max_rem, 4),
                "failure_reason": fail_reason,
                "score_drift_count": drift_count,
                "score_drift_labels": drift_labels,
            },
        )


class BlockAllFlaggedBaseline(CutSolver):
    """Cuts all flagged edges without topology optimization or cost weighting."""

    def __init__(self, tau: Optional[float] = None) -> None:
        self.tau = tau

    def solve(
        self,
        incident: Incident,
        window: GraphWindow,
        detector: Detector,
        cost_fn: Any,
    ) -> CutSet:
        if self.tau is not None:
            det = detector.predict(window)
            flagged_edge_ids = {
                e.id for e in window.edges if float(det.edge_scores.get(e.id, 0.0)) >= self.tau
            }
        else:
            flagged_edge_ids = {e.id for e in incident.target_edges}
            if not flagged_edge_ids:
                det = detector.predict(window)
                flagged_edge_ids = set(det.flagged_edges)

        cut_edges = []
        total_cost = 0.0
        for edge in window.edges:
            if edge.id in flagged_edge_ids:
                cost = float(cost_fn(edge, window)) if callable(cost_fn) else 0.0
                total_cost += cost
                cut_edges.append(
                    CutEdge(
                        edge=edge,
                        mask_weight=0.0,
                        cost=round(cost, 4),
                        virtual=False,
                    )
                )

        # Post-cut remaining verification (Decision 44)
        cut_ids = {ce.edge.id for ce in cut_edges}
        kept = [e for e in window.edges if e.id not in cut_ids]
        eff_tau = self.tau if self.tau is not None else getattr(detector, "tau", 0.50)
        drift_count = 0
        drift_labels = []
        if not kept:
            success = True
            max_rem = 0.0
            fail_reason = "None"
        else:
            nodes = sorted(list({e.src_ip for e in kept} | {e.dst_ip for e in kept}))
            nmap = {n: i for i, n in enumerate(nodes)}
            rem_win = GraphWindow(
                window_id=f"{window.window_id}_blk_check",
                start_time=window.start_time,
                end_time=window.end_time,
                node_ids=nodes,
                edge_index=[[nmap[e.src_ip] for e in kept], [nmap[e.dst_ip] for e in kept]],
                edge_attr=[
                    window.edge_attr[i] for i, e in enumerate(window.edges) if e.id not in cut_ids
                ],
                edges=kept,
            )
            rem_det = detector.predict(rem_win)
            cand_rem_scores = [
                float(rem_det.edge_scores.get(e.id, 0.0)) for e in kept if e.id in flagged_edge_ids
            ]
            max_rem = max(cand_rem_scores) if cand_rem_scores else 0.0
            # Containment: all flagged edges are blocked, so remaining candidates score < tau
            success = max_rem < eff_tau
            fail_reason = (
                "None"
                if success
                else f"Remaining candidate score {max_rem:.4f} >= tau {eff_tau:.4f}"
            )
            # Score drift: non-flagged edges that rose >= tau after the cut
            for i, e in enumerate(window.edges):
                if e.id not in cut_ids and e.id not in flagged_edge_ids:
                    sc = float(rem_det.edge_scores.get(e.id, 0.0))
                    if sc >= eff_tau:
                        drift_count += 1
                        if window.labels is not None:
                            drift_labels.append(window.labels[i])

        return CutSet(
            edges=cut_edges,
            total_cost=round(total_cost, 4),
            containment_probability=1.0 if success else 0.0,
            status_message=(
                f"Blocked {len(cut_edges)} flagged edge(s) "
                f"(success={success}, drift={drift_count})"
            ),
            metadata={
                "candidate_count": len(flagged_edge_ids),
                "initial_cut_size": 0,
                "final_cut_size": len(cut_edges),
                "iterations_used": 0,
                "max_remaining_unblocked_score": round(max_rem, 4),
                "failure_reason": fail_reason,
                "score_drift_count": drift_count,
                "score_drift_labels": drift_labels,
            },
        )
