"""Data models for GraphGuard network entities, telemetry, and containment actions."""

import ipaddress
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EdgeKey(BaseModel):
    """Deterministic, canonical identifier for a directed communication edge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    src_ip: str
    dst_ip: str
    dst_port: int = Field(ge=0, le=65535)
    proto: str

    @field_validator("src_ip", "dst_ip", mode="before")
    @classmethod
    def validate_ip(cls, value: str) -> str:
        try:
            ipaddress.ip_address(value.strip())
        except ValueError as err:
            raise ValueError(f"Invalid IP address: '{value}'") from err
        return value.strip()

    @field_validator("proto", mode="before")
    @classmethod
    def normalize_proto(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("Protocol cannot be empty.")
        return normalized

    @property
    def id(self) -> str:
        """Stable, canonical string representation of this edge."""
        return f"{self.src_ip}>{self.dst_ip}:{self.dst_port}/{self.proto}"

    def __hash__(self) -> int:
        return hash((self.src_ip, self.dst_ip, self.dst_port, self.proto))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EdgeKey):
            return False
        return (
            self.src_ip == other.src_ip
            and self.dst_ip == other.dst_ip
            and self.dst_port == other.dst_port
            and self.proto == other.proto
        )


class ActionType(str, Enum):
    """Supported network containment action types."""

    DROP = "DROP"
    RATE_LIMIT = "RATE_LIMIT"
    REDIRECT_TO_DECOY = "REDIRECT_TO_DECOY"


class CutEdge(BaseModel):
    """An individual edge selected for containment with its associated mask, cost, and action."""

    model_config = ConfigDict(extra="forbid")

    edge: EdgeKey
    mask_weight: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0)
    virtual: bool = False
    action_type: ActionType = ActionType.DROP


class CutSet(BaseModel):
    """The set of edges to intervene on, computed by a CutSolver."""

    model_config = ConfigDict(extra="forbid")

    cut_edges: List[CutEdge] = Field(default_factory=list)
    solver_name: str = "cut_solver"
    containment_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    status_message: Optional[str] = None
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            if "edges" in data and "cut_edges" not in data:
                data["cut_edges"] = data.pop("edges")
            data.pop("total_cost", None)
        return data

    @property
    def edges(self) -> List[CutEdge]:
        return self.cut_edges

    @property
    def total_cost(self) -> float:
        return sum(ce.cost for ce in self.cut_edges)


class ActionStatus(str, Enum):
    """Lifecycle statuses for containment actions."""

    PENDING = "PENDING"
    APPLIED = "APPLIED"
    REVERTED = "REVERTED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


class Action(BaseModel):
    """Action directive applied to network infrastructure."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    incident_id: Optional[str] = None
    type: ActionType
    match: EdgeKey
    status: ActionStatus = ActionStatus.PENDING
    created_at: datetime = Field(default_factory=_utc_now)
    expires_at: Optional[datetime] = None
    virtual: bool = False
    rate_limit_kbps: Optional[int] = Field(default=None, ge=1)
    decoy_ip: Optional[str] = None

    @field_validator("decoy_ip")
    @classmethod
    def validate_decoy_ip(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            ipaddress.ip_address(value.strip())
            return value.strip()
        return None


class IncidentSeverity(str, Enum):
    """Severity ratings for an incident."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentStatus(str, Enum):
    """Lifecycle statuses for an active threat incident."""

    OPEN = "OPEN"
    CONTAINED = "CONTAINED"
    ESCALATED = "ESCALATED"
    CLOSED = "CLOSED"


class Incident(BaseModel):
    """State tracking for an active or resolved containment incident."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str
    severity: IncidentSeverity = IncidentSeverity.HIGH
    target_nodes: List[str] = Field(default_factory=list)
    target_edges: List[EdgeKey] = Field(default_factory=list)
    rounds: int = Field(default=0, ge=0)
    status: IncidentStatus = IncidentStatus.OPEN
    created_at: datetime = Field(default_factory=_utc_now)
    history: List[Dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def handle_extra_incident_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            data.pop("severity", None)
        return data


class Detection(BaseModel):
    """Scoring output from GNN detector inference on a GraphWindow."""

    model_config = ConfigDict(extra="forbid")

    edge_scores: Dict[str, float] = Field(default_factory=dict)
    node_scores: Dict[str, float] = Field(default_factory=dict)
    uncertainty: Dict[str, float] = Field(default_factory=dict)
    flagged_edges: List[str] = Field(default_factory=list)
    flagged_nodes: List[str] = Field(default_factory=list)
    edge_index_map: Dict[str, int] = Field(default_factory=dict)


class GraphWindow(BaseModel):
    """Pure-Python, JSON-serializable temporal snapshot of the communication graph."""

    model_config = ConfigDict(extra="forbid")

    edge_index: List[List[int]]
    edge_attr: List[List[float]]
    node_ids: List[str]
    edges: List[EdgeKey]
    t_start: datetime = Field(default_factory=_utc_now)
    t_end: datetime = Field(default_factory=_utc_now)
    shard_id: str = "default"
    window_id: Optional[str] = None
    labels: Optional[List[int]] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_graph_window_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            if "start_time" in data and "t_start" not in data:
                data["t_start"] = data.pop("start_time")
            if "end_time" in data and "t_end" not in data:
                data["t_end"] = data.pop("end_time")
            edge_idx = data.get("edge_index")
            edges = data.get("edges")
            if (
                isinstance(edge_idx, list)
                and len(edge_idx) == 2
                and isinstance(edges, list)
                and len(edges) != 2
            ):
                srcs, dsts = edge_idx[0], edge_idx[1]
                data["edge_index"] = [[s, d] for s, d in zip(srcs, dsts)]
        return data

    @model_validator(mode="after")
    def validate_edge_consistency(self) -> "GraphWindow":
        n_indices = len(self.edge_index)
        n_attrs = len(self.edge_attr)
        n_edges = len(self.edges)
        if not (n_indices == n_attrs == n_edges):
            raise ValueError(
                f"Edge lists mismatch in GraphWindow: "
                f"len(edge_index)={n_indices}, len(edge_attr)={n_attrs}, len(edges)={n_edges}"
            )
        return self

    @property
    def start_time(self) -> datetime:
        return self.t_start

    @property
    def end_time(self) -> datetime:
        return self.t_end

    def to_pyg(self) -> Any:
        """Convert window to a PyTorch Geometric Data object with constant node features (ones).

        Initializes node features x as a column of ones (shape [num_nodes, 1]) since
        raw endpoint attributes are not embedded at window creation.
        Requires torch and torch_geometric to be installed.
        """
        try:
            import torch
            from torch_geometric.data import Data  # type: ignore[import-untyped]
        except ImportError as err:
            raise ImportError(
                "torch and torch_geometric are required to convert GraphWindow to PyG Data."
            ) from err

        num_nodes = len(self.node_ids)
        x = torch.ones((num_nodes, 1), dtype=torch.float)

        if self.edge_index:
            edge_index_tensor = torch.tensor(self.edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(self.edge_attr, dtype=torch.float)
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, 0), dtype=torch.float)

        return Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)


class FlowRecord(BaseModel):
    """Raw flow metadata ingested from network interfaces."""

    model_config = ConfigDict(extra="forbid")

    src_ip: str
    dst_ip: str
    src_port: int = Field(ge=0, le=65535)
    dst_port: int = Field(ge=0, le=65535)
    proto: str
    bytes_count: int = Field(ge=0)
    packets_count: int = Field(ge=0)
    duration: float = Field(ge=0.0)
    tcp_flags: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=_utc_now)

    @field_validator("src_ip", "dst_ip", mode="before")
    @classmethod
    def validate_ip(cls, value: str) -> str:
        ipaddress.ip_address(value.strip())
        return value.strip()

    @field_validator("proto", mode="before")
    @classmethod
    def normalize_proto(cls, value: str) -> str:
        return value.strip().lower()

    @property
    def edge_key(self) -> EdgeKey:
        return EdgeKey(
            src_ip=self.src_ip,
            dst_ip=self.dst_ip,
            dst_port=self.dst_port,
            proto=self.proto,
        )


class Verdict(str, Enum):
    """Verification outcome produced by Verifier after an intervention round."""

    CONTAINED = "CONTAINED"
    UNCONTAINED = "UNCONTAINED"
    ESCALATED = "ESCALATED"


class VerificationRecord(BaseModel):
    """Detailed record of an intervention verification round."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str
    round_num: int = Field(ge=1)
    verdict: Verdict
    scores_before: Dict[str, float] = Field(default_factory=dict)
    scores_after: Dict[str, float] = Field(default_factory=dict)
    edges_cut: List[str] = Field(default_factory=list)
    rules_installed: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utc_now)


class ReactionReport(BaseModel):
    """Evaluation of behavioral response post-containment action (N6)."""

    model_config = ConfigDict(extra="forbid")

    source: str
    evasion_score: float = Field(ge=0.0, le=1.0)
    retries_count: int = Field(ge=0)
    new_destinations: List[str] = Field(default_factory=list)
    port_switching: bool = False
    fanout_delta: float = 0.0
    decoy_traffic_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class AuditEvent(BaseModel):
    """Immutable audit trail event for compliance and forensic review."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    actor: str
    action: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utc_now)
