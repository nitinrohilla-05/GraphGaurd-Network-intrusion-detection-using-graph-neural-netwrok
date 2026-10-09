"""Unit tests for classification metrics, binary metrics, and hub dependence."""

import numpy as np
import pandas as pd

from graphguard.detect.egraphsage import EGraphSAGEDetector, EGraphSAGENet
from graphguard.eval.metrics import (
    compute_binary_metrics,
    compute_classification_metrics,
    evaluate_hub_dependence,
)
from graphguard.graph.features import NetFlowFeatureExtractor


def test_compute_classification_metrics() -> None:
    """Per-class F1, macro-F1, and weighted-F1 calculations."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 2, 2])
    class_names = ["Benign", "PortScan", "DDoS"]

    res = compute_classification_metrics(y_true, y_pred, class_names)
    assert "per_class" in res
    assert "Benign" in res["per_class"]
    assert "macro_f1" in res
    assert 0.0 <= res["macro_f1"] <= 1.0


def test_compute_binary_metrics() -> None:
    """Binary precision, recall, and F1 calculations based on tau."""
    y_true_binary = np.array([0, 1, 1, 0])
    attack_scores = np.array([0.1, 0.9, 0.8, 0.2])

    metrics = compute_binary_metrics(y_true_binary, attack_scores, tau=0.50)
    assert metrics["binary_precision"] == 1.0
    assert metrics["binary_recall"] == 1.0
    assert metrics["binary_f1"] == 1.0


def test_evaluate_hub_dependence() -> None:
    """Per-flow source randomization hub-dependence check executes without error."""
    df = pd.DataFrame(
        {
            "IPV4_SRC_ADDR": ["10.0.0.1", "10.0.0.1"],
            "IPV4_DST_ADDR": ["10.0.0.2", "10.0.0.3"],
            "L4_DST_PORT": [80, 443],
            "PROTOCOL": [6, 6],
            "IN_BYTES": [500.0, 1000.0],
            "Label": [0, 1],
        }
    )
    extractor = NetFlowFeatureExtractor()
    extractor.fit(df)

    net = EGraphSAGENet(edge_dim=len(extractor.feature_names), num_classes=2, hidden_dim=16)
    detector = EGraphSAGEDetector(model=net, classes=["Benign", "Attack"])

    hub_res = evaluate_hub_dependence(df, extractor, detector, original_f1=0.85)
    assert "original_f1" in hub_res
    assert "randomized_f1" in hub_res
    assert "f1_delta" in hub_res
