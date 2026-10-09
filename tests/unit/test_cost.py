"""Unit tests for asset criticality, role inference, and edge cost computation."""

import pandas as pd
import pytest

from graphguard.containment.cost import EdgeCostEvaluator, infer_asset_roles_from_flows
from graphguard.core.models import EdgeKey, GraphWindow


def test_infer_asset_roles_from_flows(tmp_path: pytest.TempPathFactory) -> None:
    """Validate inference of DNS, Database, and Gateway roles from traffic patterns."""
    df = pd.DataFrame(
        {
            "IPV4_SRC_ADDR": [
                "192.168.1.10",
                "192.168.1.11",
                "192.168.1.10",
                "192.168.1.10",
            ]
            * 3,
            "IPV4_DST_ADDR": [
                "192.168.1.2",  # DNS
                "192.168.1.50",  # DB
                "192.168.1.1",  # Gateway
                "192.168.1.50",  # DB
            ]
            * 3,
            "L4_DST_PORT": [53, 3306, 80, 5432] * 3,
            "PROTOCOL": [17, 6, 6, 6] * 3,
        }
    )

    out_file = str(tmp_path / "assets_test.yaml")
    roles = infer_asset_roles_from_flows(df, output_path=out_file)

    assert "192.168.1.2" in roles
    assert roles["192.168.1.2"]["role"] == "dns"
    assert roles["192.168.1.2"]["criticality"] == 0.95

    assert "192.168.1.50" in roles
    assert roles["192.168.1.50"]["role"] == "database"
    assert roles["192.168.1.50"]["criticality"] == 0.85

    assert "192.168.1.10" in roles
    assert roles["192.168.1.10"]["role"] == "gateway"  # Top communicating host with 12 flows


def test_edge_cost_evaluator_formula() -> None:
    """Validate cost(e) = alpha * criticality(src, dst) + beta * volume."""
    evaluator = EdgeCostEvaluator(alpha=0.7, beta=0.3, default_criticality=0.2)
    evaluator.node_criticality["192.168.1.1"] = 1.0  # Gateway
    evaluator.node_criticality["192.168.1.10"] = 0.3  # Workstation

    edge = EdgeKey(
        src_ip="192.168.1.10",
        dst_ip="192.168.1.1",
        dst_port=80,
        proto="tcp",
    )

    window = GraphWindow(
        window_id="win_test",
        node_ids=["192.168.1.10", "192.168.1.1"],
        edge_index=[[0, 1]],
        edge_attr=[[1.5, 0.5]],
        edges=[edge],
    )

    cost = evaluator.compute_cost(edge, window)
    # crit_max is max(0.3, 1.0) = 1.0 -> alpha * 1.0 = 0.7
    # volume > 0 -> beta * vol_norm > 0
    assert 0.70 <= cost <= 1.0
