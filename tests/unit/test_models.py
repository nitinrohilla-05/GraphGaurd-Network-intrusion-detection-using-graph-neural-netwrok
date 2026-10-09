"""Unit tests for GraphGuard core data models."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from graphguard.core.models import (
    Action,
    ActionStatus,
    ActionType,
    AuditEvent,
    CutEdge,
    CutSet,
    Detection,
    EdgeKey,
    FlowRecord,
    GraphWindow,
    Incident,
    IncidentStatus,
    ReactionReport,
)


def test_edge_key_stability_and_ports() -> None:
    """EdgeKey ID must be stable, deterministic, and distinguish different ports."""
    e1 = EdgeKey(src_ip="192.168.1.10", dst_ip="10.0.0.5", dst_port=80, proto="tcp")
    e2 = EdgeKey(src_ip="192.168.1.10", dst_ip="10.0.0.5", dst_port=443, proto="tcp")
    e3 = EdgeKey(src_ip="192.168.1.10", dst_ip="10.0.0.5", dst_port=80, proto="TCP")

    assert e1.id == "192.168.1.10>10.0.0.5:80/tcp"
    assert e2.id == "192.168.1.10>10.0.0.5:443/tcp"
    assert e1 != e2
    assert e1.id != e2.id
    assert e1 == e3
    assert e1.id == e3.id
    assert hash(e1) == hash(e3)

    # Hashable in sets
    edge_set = {e1, e2, e3}
    assert len(edge_set) == 2


def test_edge_key_ip_validation() -> None:
    """Invalid IP addresses must be rejected with ValueError."""
    with pytest.raises(ValidationError):
        EdgeKey(src_ip="not_an_ip", dst_ip="10.0.0.1", dst_port=80, proto="tcp")

    with pytest.raises(ValidationError):
        EdgeKey(src_ip="192.168.1.1", dst_ip="999.999.999.999", dst_port=80, proto="tcp")

    with pytest.raises(ValidationError):
        EdgeKey(src_ip="192.168.1.1", dst_ip="10.0.0.1", dst_port=70000, proto="tcp")


def test_cutset_json_roundtrip() -> None:
    """CutSet must cleanly serialize and deserialize to JSON without tuple keys."""
    k1 = EdgeKey(src_ip="192.168.1.10", dst_ip="10.0.0.5", dst_port=80, proto="tcp")
    k2 = EdgeKey(src_ip="192.168.1.11", dst_ip="10.0.0.5", dst_port=22, proto="tcp")

    ce1 = CutEdge(edge=k1, mask_weight=0.15, cost=0.85, virtual=False)
    ce2 = CutEdge(edge=k2, mask_weight=0.45, cost=0.30, virtual=True)

    cut_set = CutSet(cut_edges=[ce1, ce2], solver_name="counterfactual_opt")
    raw_json = cut_set.model_dump_json()

    # Validate JSON string format
    parsed = json.loads(raw_json)
    assert len(parsed["cut_edges"]) == 2
    assert parsed["solver_name"] == "counterfactual_opt"

    # Round trip
    loaded = CutSet.model_validate_json(raw_json)
    assert len(loaded.cut_edges) == 2
    assert loaded.cut_edges[0].edge == k1
    assert loaded.cut_edges[1].virtual is True


def test_action_json_roundtrip_and_enums() -> None:
    """Action model serialization, enum statuses, and TTL field roundtrip."""
    edge = EdgeKey(src_ip="10.0.1.5", dst_ip="10.0.0.1", dst_port=53, proto="udp")
    now = datetime.now(timezone.utc)
    expiry = now + timedelta(seconds=60)

    action = Action(
        action_id="act_001",
        incident_id="inc_42",
        type=ActionType.REDIRECT_TO_DECOY,
        match=edge,
        status=ActionStatus.APPLIED,
        created_at=now,
        expires_at=expiry,
        decoy_ip="10.0.0.99",
    )

    raw_json = action.model_dump_json()
    loaded = Action.model_validate_json(raw_json)

    assert loaded.action_id == "act_001"
    assert loaded.incident_id == "inc_42"
    assert loaded.type == ActionType.REDIRECT_TO_DECOY
    assert loaded.status == ActionStatus.APPLIED
    assert loaded.match == edge
    assert loaded.decoy_ip == "10.0.0.99"


def test_incident_status_enum() -> None:
    """Incident status transitions and validation."""
    inc = Incident(
        incident_id="inc_100",
        target_nodes=["10.0.0.10"],
        rounds=1,
        status=IncidentStatus.OPEN,
    )
    assert inc.status == IncidentStatus.OPEN
    inc.status = IncidentStatus.CONTAINED
    assert inc.status == IncidentStatus.CONTAINED


def test_graph_window_consistency_and_validation() -> None:
    """GraphWindow model validator ensures edge list lengths match."""
    now = datetime.now(timezone.utc)
    k1 = EdgeKey(src_ip="10.0.0.2", dst_ip="10.0.0.10", dst_port=80, proto="tcp")
    k2 = EdgeKey(src_ip="10.0.0.3", dst_ip="10.0.0.10", dst_port=80, proto="tcp")

    # Mismatched lengths should fail validation
    with pytest.raises(ValidationError):
        GraphWindow(
            edge_index=[[0, 1]],  # length 1
            edge_attr=[[1.0], [2.0]],  # length 2
            node_ids=["10.0.0.2", "10.0.0.3", "10.0.0.10"],
            edges=[k1, k2],  # length 2
            t_start=now,
            t_end=now + timedelta(seconds=30),
        )

    # Valid window
    window = GraphWindow(
        edge_index=[[0, 2], [1, 2]],
        edge_attr=[[1.0, 50.0], [2.0, 100.0]],
        node_ids=["10.0.0.2", "10.0.0.3", "10.0.0.10"],
        edges=[k1, k2],
        t_start=now,
        t_end=now + timedelta(seconds=30),
        shard_id="subnet-1",
    )
    assert len(window.edges) == 2
    raw = window.model_dump_json()
    reloaded = GraphWindow.model_validate_json(raw)
    assert len(reloaded.edges) == 2


def test_graph_window_to_pyg_docstring_and_import() -> None:
    """to_pyg() documentation and execution check."""
    now = datetime.now(timezone.utc)
    window = GraphWindow(
        edge_index=[],
        edge_attr=[],
        node_ids=["10.0.0.1"],
        edges=[],
        t_start=now,
        t_end=now + timedelta(seconds=30),
    )
    assert "constant node features (ones" in (GraphWindow.to_pyg.__doc__ or "")

    try:
        data = window.to_pyg()
        assert data.x.shape == (1, 1)
    except ImportError as err:
        assert "torch" in str(err)


def test_flow_record_and_edge_key_property() -> None:
    """FlowRecord produces correct EdgeKey and validates data types."""
    record = FlowRecord(
        src_ip="192.168.0.50",
        dst_ip="10.0.0.1",
        src_port=54321,
        dst_port=53,
        proto="UDP",
        bytes_count=120,
        packets_count=2,
        duration=0.015,
    )
    assert record.proto == "udp"
    assert record.edge_key == EdgeKey(
        src_ip="192.168.0.50", dst_ip="10.0.0.1", dst_port=53, proto="udp"
    )


def test_detection_and_reaction_report() -> None:
    """Detection, ReactionReport, and AuditEvent schemas."""
    e_id = "10.0.0.2>10.0.0.10:80/tcp"
    det = Detection(
        edge_scores={e_id: 0.95},
        node_scores={"10.0.0.2": 0.88},
        uncertainty={e_id: 0.05},
        flagged_edges=[e_id],
        flagged_nodes=["10.0.0.2"],
    )
    assert det.edge_scores[e_id] == 0.95

    report = ReactionReport(
        source="10.0.0.2",
        evasion_score=0.82,
        retries_count=5,
        new_destinations=["10.0.0.11", "10.0.0.12"],
        port_switching=True,
        fanout_delta=2.0,
        decoy_traffic_ratio=0.10,
    )
    assert report.evasion_score == 0.82
    assert report.port_switching is True

    audit = AuditEvent(
        event_id="evt_01",
        actor="operator@graphguard",
        action="APPROVAL",
        details={"incident_id": "inc_1"},
    )
    assert audit.action == "APPROVAL"
