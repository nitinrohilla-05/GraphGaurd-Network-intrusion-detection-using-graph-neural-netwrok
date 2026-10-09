"""Asset criticality scoring, data-driven role inference, and edge cost computation (N1)."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from graphguard.core.models import EdgeKey, GraphWindow

logger = logging.getLogger(__name__)

DEFAULT_ROLE_WEIGHTS = {
    "gateway": 1.0,
    "dns": 0.95,
    "domain_controller": 0.95,
    "database": 0.85,
    "dhcp": 0.80,
    "internal_server": 0.60,
    "workstation": 0.30,
    "decoy": 0.05,
    "default": 0.20,
}


def infer_asset_roles_from_flows(
    df: pd.DataFrame,
    output_path: str = "config/assets_inferred.yaml",
) -> Dict[str, Dict[str, Any]]:
    """Infer asset roles directly from network flow traffic patterns (Decision 33).

    Rules:
    - Hosts receiving most traffic on port 53 -> 'dns'
    - Hosts receiving traffic on database ports (3306, 5432, 1433) -> 'database'
    - Top-degree communicating host -> 'gateway'
    """
    lookup = {c.lower().strip(): c for c in df.columns}
    dst_col = lookup.get("ipv4_dst_addr") or lookup.get("destination ip") or lookup.get("dst_ip")
    dport_col = (
        lookup.get("l4_dst_port") or lookup.get("destination port") or lookup.get("dst_port")
    )
    src_col = lookup.get("ipv4_src_addr") or lookup.get("source ip") or lookup.get("src_ip")

    inferred_nodes: Dict[str, Dict[str, Any]] = {}

    if not dst_col or not dport_col or len(df) == 0:
        return inferred_nodes

    df_clean = df.copy()
    df_clean[dport_col] = pd.to_numeric(df_clean[dport_col], errors="coerce").fillna(0).astype(int)

    # 1. DNS inference: destination port 53
    dns_flows = df_clean[df_clean[dport_col] == 53]
    if len(dns_flows) > 0:
        top_dns = dns_flows[dst_col].value_counts()
        for ip, count in top_dns.head(3).items():
            inferred_nodes[str(ip)] = {
                "role": "dns",
                "criticality": DEFAULT_ROLE_WEIGHTS["dns"],
                "reason": f"Received {count} DNS (port 53) flows",
            }

    # 2. Database inference: ports 3306 (MySQL), 5432 (Postgres), 1433 (MSSQL)
    db_ports = {3306, 5432, 1433}
    db_flows = df_clean[df_clean[dport_col].isin(db_ports)]
    if len(db_flows) > 0:
        top_dbs = db_flows[dst_col].value_counts()
        for ip, count in top_dbs.head(5).items():
            if count >= 5 and str(ip) not in inferred_nodes:
                inferred_nodes[str(ip)] = {
                    "role": "database",
                    "criticality": DEFAULT_ROLE_WEIGHTS["database"],
                    "reason": f"Received {count} DB connection flows",
                }

    # 3. Gateway inference: top-degree host communicating across both src and dst
    if src_col:
        degree_series = pd.concat([df_clean[src_col], df_clean[dst_col]]).value_counts()
        for ip, total_flows in degree_series.items():
            ip_str = str(ip)
            if ip_str not in inferred_nodes:
                inferred_nodes[ip_str] = {
                    "role": "gateway",
                    "criticality": DEFAULT_ROLE_WEIGHTS["gateway"],
                    "reason": f"Top communicating network host with {total_flows} flows",
                }
                break

    # Save to yaml
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    export_payload = {
        "cost_weights": {"alpha": 0.7, "beta": 0.3},
        "default_criticality": DEFAULT_ROLE_WEIGHTS["default"],
        "nodes": inferred_nodes,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(export_payload, f, sort_keys=False)

    logger.info(f"Inferred {len(inferred_nodes)} asset roles and saved to {output_path}")
    return inferred_nodes


class EdgeCostEvaluator:
    """Evaluates cost(e) = α * criticality(src, dst) + β * normalised_benign_volume(e)."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        alpha: float = 0.7,
        beta: float = 0.3,
        default_criticality: float = 0.20,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.default_criticality = default_criticality
        self.node_criticality: Dict[str, float] = {}

        # Load configs if available
        candidates = [
            config_path,
            "config/assets_inferred.yaml",
            "config/assets.yaml",
        ]
        for p in candidates:
            if p and Path(p).exists():
                self._load_from_yaml(p)
                break

    def _load_from_yaml(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                weights = data.get("cost_weights", {})
                self.alpha = float(weights.get("alpha", self.alpha))
                self.beta = float(weights.get("beta", self.beta))
                self.default_criticality = float(
                    data.get("default_criticality", self.default_criticality)
                )

                nodes_dict = data.get("nodes", {})
                for ip, info in nodes_dict.items():
                    if isinstance(info, dict) and "criticality" in info:
                        self.node_criticality[str(ip)] = float(info["criticality"])
                    elif isinstance(info, (int, float)):
                        self.node_criticality[str(ip)] = float(info)
        except Exception as err:
            logger.warning(f"Could not load asset config {path}: {err}")

    def get_node_criticality(self, ip: str) -> float:
        return self.node_criticality.get(ip, self.default_criticality)

    def compute_cost(self, edge: EdgeKey, window: GraphWindow) -> float:
        """Calculate composite cost for a given edge within a graph window."""
        c_src = self.get_node_criticality(edge.src_ip)
        c_dst = self.get_node_criticality(edge.dst_ip)
        crit_max = max(c_src, c_dst)

        # Approximate volume from edge features if present
        # In standardized features, index 0 is typically bytes
        vol_norm = 0.5  # default neutral volume
        if window.edges:
            try:
                idx = window.edges.index(edge)
                if idx < len(window.edge_attr) and len(window.edge_attr[idx]) > 0:
                    raw_val = abs(float(window.edge_attr[idx][0]))
                    # Sigmoid-normalized to [0, 1]
                    vol_norm = float(1.0 / (1.0 + np.exp(-raw_val / 2.0)))
            except (ValueError, IndexError):
                pass

        total_cost = self.alpha * crit_max + self.beta * vol_norm
        return float(np.clip(total_cost, 0.0, 1.0))

    def __call__(self, edge: EdgeKey, window: GraphWindow) -> float:
        return self.compute_cost(edge, window)
