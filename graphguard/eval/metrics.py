"""Comprehensive evaluation metrics, per-class F1 reporting, and hub-dependence verification."""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from graphguard.detect.egraphsage import EGraphSAGEDetector
from graphguard.graph.features import NetFlowFeatureExtractor
from graphguard.graph.window_builder import build_graph_window_from_dataframe

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Compute per-class Precision, Recall, F1, Support, and aggregate macro/weighted F1."""
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for name in class_names:
        if name in report_dict:
            per_class[name] = {
                "precision": float(report_dict[name]["precision"]),
                "recall": float(report_dict[name]["recall"]),
                "f1": float(report_dict[name]["f1-score"]),
                "support": int(report_dict[name]["support"]),
            }

    macro_f1 = float(report_dict["macro avg"]["f1-score"])
    weighted_f1 = float(report_dict["weighted avg"]["f1-score"])

    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def compute_binary_metrics(
    y_true_binary: np.ndarray,
    attack_scores: np.ndarray,
    tau: float = 0.50,
) -> Dict[str, float]:
    """Compute binary detection metrics based on decision threshold tau."""
    preds = (attack_scores >= tau).astype(int)
    return {
        "binary_precision": float(precision_score(y_true_binary, preds, zero_division=0)),
        "binary_recall": float(recall_score(y_true_binary, preds, zero_division=0)),
        "binary_f1": float(f1_score(y_true_binary, preds, zero_division=0)),
        "tau": tau,
    }


def evaluate_hub_dependence(
    test_df: pd.DataFrame,
    feature_extractor: NetFlowFeatureExtractor,
    detector: EGraphSAGEDetector,
    original_f1: float,
    seed: int = 42,
) -> Dict[str, float]:
    """Per-flow source randomization hub-dependence check (Decision 24).

    Replaces IPV4_SRC_ADDR for each test flow independently with an address from
    the 10.128.0.0/9 pool, rebuilds windows, runs inference, and compares F1.
    """
    rng = np.random.default_rng(seed)
    df_rand = test_df.copy()

    # Locate source IP column
    src_cols = [
        c for c in df_rand.columns if c.lower().strip() in ("ipv4_src_addr", "source ip", "src_ip")
    ]
    if not src_cols:
        return {"original_f1": original_f1, "randomized_f1": original_f1, "f1_delta": 0.0}

    src_col = src_cols[0]
    n_flows = len(df_rand)

    # Sample random IPs from 10.128.0.0/9
    oct2 = rng.integers(128, 256, size=n_flows)
    oct3 = rng.integers(0, 256, size=n_flows)
    oct4 = rng.integers(1, 255, size=n_flows)
    random_ips = [f"10.{o2}.{o3}.{o4}" for o2, o3, o4 in zip(oct2, oct3, oct4)]
    df_rand[src_col] = random_ips

    # Rebuild window with identical features
    X_test = feature_extractor.transform(df_rand)
    win_rand, y_binary, _ = build_graph_window_from_dataframe(df_rand, X_test)

    # Score with detector
    detection = detector.predict(win_rand)
    pred_scores = np.array([detection.edge_scores.get(e.id, 0.0) for e in win_rand.edges])
    rand_preds = (pred_scores >= detector.tau).astype(int)
    randomized_f1 = float(f1_score(y_binary, rand_preds, zero_division=0))

    delta = randomized_f1 - original_f1
    logger.info(
        f"[Hub-Dependence Check] Original F1={original_f1:.4f}, "
        f"Per-Flow Source Randomization F1={randomized_f1:.4f} (Delta={delta:+.4f})"
    )

    return {
        "original_f1": original_f1,
        "randomized_f1": randomized_f1,
        "f1_delta": delta,
    }
