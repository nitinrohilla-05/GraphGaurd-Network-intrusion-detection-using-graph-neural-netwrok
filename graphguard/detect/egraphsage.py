"""E-GraphSAGE model with edge feature message passing and Detector ABC implementation."""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from graphguard.core.interfaces import Detector
from graphguard.core.models import Detection, GraphWindow


class EdgeSAGEConv(MessagePassing):
    """GraphSAGE conv layer aggregating node representations fused with edge attributes.

    Implements mask-weighted mean aggregation: sum(M_e * m_e) / (sum(M_e) + eps) (Decision 31).
    When M_e -> 0, the edge's contribution to node aggregation is completely eliminated.
    """

    def __init__(
        self,
        in_channels: int,
        edge_channels: int,
        out_channels: int,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(aggr=None)
        self.eps = eps
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_channels, out_channels),
            nn.ReLU(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            zeros = torch.zeros(
                (x.size(0), self.msg_mlp[0].out_features),
                device=x.device,
                dtype=x.dtype,
            )
            return self.update_mlp(torch.cat([x, zeros], dim=-1))

        if edge_mask is None:
            edge_mask = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)

        j = edge_index[0]  # source node
        i = edge_index[1]  # target node
        concat = torch.cat([x[j], edge_attr], dim=-1)
        msg = self.msg_mlp(concat)

        mask_weights = edge_mask.view(-1, 1)
        weighted_msg = mask_weights * msg

        from torch_geometric.utils import scatter

        numer = scatter(weighted_msg, i, dim=0, dim_size=x.size(0), reduce="sum")
        denom = scatter(mask_weights, i, dim=0, dim_size=x.size(0), reduce="sum")
        aggr_out = numer / (denom + self.eps)

        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class EGraphSAGENet(nn.Module):
    """2-layer E-GraphSAGE network for multi-class edge classification."""

    def __init__(
        self,
        edge_dim: int,
        num_classes: int,
        node_dim: int = 1,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.edge_dim = edge_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.conv1 = EdgeSAGEConv(node_dim, edge_dim, hidden_dim)
        self.conv2 = EdgeSAGEConv(hidden_dim, edge_dim, hidden_dim)

        # Edge classifier head: fuses [src_node, dst_node, edge_features]
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning edge classification logits [num_edges, num_classes]."""
        if edge_index.numel() == 0:
            return torch.empty((0, self.num_classes), device=x.device)

        h1 = self.conv1(x, edge_index, edge_attr, edge_mask=edge_mask)
        h1 = F.dropout(h1, p=self.dropout_rate, training=self.training)

        h2 = self.conv2(h1, edge_index, edge_attr, edge_mask=edge_mask)

        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h2[src], h2[dst], edge_attr], dim=-1)
        logits = self.edge_classifier(edge_repr)
        return logits


class EGraphSAGEDetector(Detector):
    """Detector implementation wrapping EGraphSAGENet with calibration and open-set scoring."""

    def __init__(
        self,
        model: EGraphSAGENet,
        classes: List[str],
        benign_class_idx: int = 0,
        tau: float = 0.50,
        energy_threshold: float = -5.0,
        mc_dropout: bool = False,
        mc_samples: int = 5,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.classes = classes
        self.benign_class_idx = benign_class_idx
        self.tau = tau
        self.energy_threshold = energy_threshold
        self.mc_dropout = mc_dropout
        self.mc_samples = mc_samples
        self.device = torch.device(device)
        self.model.to(self.device)

    def predict(self, window: GraphWindow) -> Detection:
        """Score graph window returning edge scores (by EdgeKey.id), node scores, uncertainty."""
        self.model.eval()

        if not window.edges or not window.edge_index:
            return Detection(
                edge_scores={},
                node_scores={nid: 0.0 for nid in window.node_ids},
                uncertainty={},
                flagged_edges=[],
                flagged_nodes=[],
                edge_index_map={},
            )

        pyg_data = window.to_pyg().to(self.device)

        with torch.no_grad():
            if self.mc_dropout:
                # Enable dropout during inference for MC-dropout variance
                self.model.train()
                samples = []
                for _ in range(self.mc_samples):
                    logits_s = self.model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
                    probs_s = F.softmax(logits_s, dim=-1)
                    samples.append(probs_s.unsqueeze(0))
                self.model.eval()
                stacked = torch.cat(samples, dim=0)  # [mc_samples, num_edges, num_classes]
                mean_probs = stacked.mean(dim=0)
                variances = stacked.var(dim=0).mean(dim=-1).cpu().numpy()  # [num_edges]
                probs = mean_probs.cpu().numpy()
            else:
                logits = self.model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                variances = np.zeros(len(window.edges))

        edge_scores: Dict[str, float] = {}
        uncertainty: Dict[str, float] = {}
        flagged_edges: List[str] = []
        edge_index_map: Dict[str, int] = {}
        node_incident_scores: Dict[str, List[float]] = {nid: [] for nid in window.node_ids}

        for i, edge in enumerate(window.edges):
            e_id = edge.id
            edge_index_map[e_id] = i

            # Binary attack score = 1 - P(benign) per Decision 26
            p_benign = float(probs[i, self.benign_class_idx])
            attack_score = float(np.clip(1.0 - p_benign, 0.0, 1.0))

            edge_scores[e_id] = attack_score
            uncertainty[e_id] = float(variances[i])

            if attack_score >= self.tau:
                flagged_edges.append(e_id)

            # Record incident edge scores for node aggregation
            if edge.src_ip in node_incident_scores:
                node_incident_scores[edge.src_ip].append(attack_score)
            if edge.dst_ip in node_incident_scores:
                node_incident_scores[edge.dst_ip].append(attack_score)

        node_scores: Dict[str, float] = {}
        flagged_nodes: List[str] = []
        for nid, scores in node_incident_scores.items():
            max_score = float(max(scores)) if scores else 0.0
            node_scores[nid] = max_score
            if max_score >= self.tau:
                flagged_nodes.append(nid)

        return Detection(
            edge_scores=edge_scores,
            node_scores=node_scores,
            uncertainty=uncertainty,
            flagged_edges=flagged_edges,
            flagged_nodes=flagged_nodes,
            edge_index_map=edge_index_map,
        )
