"""Unit tests for E-GraphSAGE model architecture and Detector predict interface."""

from datetime import datetime, timezone

import torch

from graphguard.core.models import EdgeKey, GraphWindow
from graphguard.detect.egraphsage import EGraphSAGEDetector, EGraphSAGENet


def test_egraphsage_net_forward_pass() -> None:
    """EGraphSAGENet produces output tensor with shape [num_edges, num_classes]."""
    num_nodes = 4
    num_edges = 5
    edge_dim = 8
    num_classes = 3

    x = torch.ones((num_nodes, 1))
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long)
    edge_attr = torch.randn((num_edges, edge_dim))

    net = EGraphSAGENet(edge_dim=edge_dim, num_classes=num_classes, hidden_dim=16)
    net.eval()
    with torch.no_grad():
        out = net(x, edge_index, edge_attr)

    assert out.shape == (num_edges, num_classes)


def test_detector_predict_interface_compliance() -> None:
    """EGraphSAGEDetector scores all edges keyed by EdgeKey.id and calculates binary score."""
    edge_dim = 4
    num_classes = 2
    net = EGraphSAGENet(edge_dim=edge_dim, num_classes=num_classes, hidden_dim=16)

    k1 = EdgeKey(src_ip="10.0.0.1", dst_ip="10.0.0.2", dst_port=80, proto="tcp")
    k2 = EdgeKey(src_ip="10.0.0.2", dst_ip="10.0.0.3", dst_port=443, proto="tcp")

    now = datetime.now(timezone.utc)
    window = GraphWindow(
        edge_index=[[0, 1], [1, 2]],
        edge_attr=[[0.5] * edge_dim, [1.5] * edge_dim],
        node_ids=["10.0.0.1", "10.0.0.2", "10.0.0.3"],
        edges=[k1, k2],
        t_start=now,
        t_end=now,
    )

    detector = EGraphSAGEDetector(
        model=net,
        classes=["Benign", "PortScan"],
        benign_class_idx=0,
        tau=0.40,
    )

    detection = detector.predict(window)

    assert k1.id in detection.edge_scores
    assert k2.id in detection.edge_scores
    assert 0.0 <= detection.edge_scores[k1.id] <= 1.0
    assert "10.0.0.1" in detection.node_scores
    assert len(detection.edge_index_map) == 2
