"""Counterfactual Minimal Cut Solver (N1) and Greedy Cut Solver (Decisions 29, 30, 31)."""

import logging
import time
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graphguard.containment.subgraph import SubgraphRegion, extract_khop_subgraph
from graphguard.core.interfaces import CutSolver, Detector
from graphguard.core.models import (
    ActionType,
    CutEdge,
    CutSet,
    EdgeKey,
    GraphWindow,
    Incident,
)

logger = logging.getLogger(__name__)


class CounterfactualCutSolver(CutSolver):
    """N1: Cost-aware counterfactual minimal cut solver.

    Optimizes soft edge masks M_e in [0, 1] over candidate edges within a k-hop subgraph.
    Enforces candidate edge restrictions (Decision 29), mask-weighted aggregation
    equivalence (Decision 31), and hard-removal verification with iterative remediation
    (Decision 30).
    """

    def __init__(
        self,
        tau_low: float = 0.30,
        tau: float = 0.50,
        rate_limit_factor: float = 0.3,
        lambda1: float = 1.0,  # cost penalty weight
        lambda2: float = 0.5,  # sparsity penalty weight
        lambda_residual_risk: float = 5.0,  # penalty for rate-limited attack edges
        max_steps: int = 150,
        learning_rate: float = 0.05,
        k_hops: int = 2,
        remediation_max_iters: int = 10,
        drop_threshold: float = 0.35,
        rate_limit_threshold: float = 0.70,
        device: str = "cpu",
    ) -> None:
        self.tau_low = tau_low
        self.tau = tau
        self.rate_limit_factor = rate_limit_factor
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_residual_risk = lambda_residual_risk
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.k_hops = k_hops
        self.remediation_max_iters = remediation_max_iters
        self.drop_threshold = drop_threshold
        self.rate_limit_threshold = rate_limit_threshold
        self.device = torch.device(device)

    def solve(
        self,
        incident: Incident,
        window: GraphWindow,
        detector: Detector,
        cost_fn: Callable[[EdgeKey, GraphWindow], float],
    ) -> CutSet:
        """Compute cost-aware minimal cut set to contain flagged malicious subgraph."""
        start_time = time.perf_counter()

        # 1. Identify flagged target edges
        flagged_edge_ids: List[str] = [e.id for e in incident.target_edges]
        if not flagged_edge_ids:
            initial_det = detector.predict(window)
            flagged_edge_ids = initial_det.flagged_edges
        else:
            initial_det = detector.predict(window)

        if not flagged_edge_ids or not window.edges:
            return CutSet(
                edges=[],
                total_cost=0.0,
                containment_probability=1.0,
                status_message="No flagged edges to contain",
            )

        # 2. Extract k-hop induced subgraph with candidate restriction (Decision 29)
        sub_region = extract_khop_subgraph(
            window=window,
            flagged_edge_ids=flagged_edge_ids,
            detection=initial_det,
            k_hops=self.k_hops,
            tau_low=self.tau_low,
        )

        sub_window = sub_region.window
        num_sub_edges = len(sub_window.edges)
        candidate_indices = sub_region.candidate_edge_indices

        if not candidate_indices or num_sub_edges == 0:
            return CutSet(
                edges=[],
                total_cost=0.0,
                containment_probability=0.0,
                status_message="No candidate edges identified in subgraph",
            )

        # Precompute edge costs for all subgraph edges
        edge_costs = np.array([cost_fn(e, window) for e in sub_window.edges], dtype=np.float32)

        # 3. Optimize soft mask over candidate edges
        best_mask = self._optimize_mask(
            sub_region=sub_region,
            detector=detector,
            candidate_indices=candidate_indices,
            edge_costs=edge_costs,
        )

        # 4. Continuous action discretization (Decision 48 / N1+N3):
        # M_e < drop_threshold -> DROP (mask=0.0, cost=1.0*cost(e))
        # drop_threshold <= M_e < rate_limit_threshold -> RATE_LIMIT
        # M_e >= rate_limit_threshold -> ALLOW (kept)
        drop_indices: Set[int] = set()
        rate_limit_indices: Set[int] = set()
        for idx in candidate_indices:
            m_val = float(best_mask[idx])
            if m_val < self.drop_threshold:
                drop_indices.add(idx)
            elif m_val < self.rate_limit_threshold:
                rate_limit_indices.add(idx)

        # 5. Success condition & Remediation (Decisions 30 & 48)
        # Condition: remaining edges with score >= tau must be DROP or RATE_LIMIT.
        (
            final_drop_indices,
            final_rate_limit_indices,
            success,
            iters_used,
            max_rem,
            fail_reason,
            unmasked_attacks,
            drifted_benign,
        ) = self._verify_and_remediate(
            sub_window=sub_window,
            initial_drop_indices=drop_indices,
            initial_rate_limit_indices=rate_limit_indices,
            candidate_indices=candidate_indices,
            detector=detector,
        )

        # 6. Build CutSet with explicit action types and costs
        cut_edges: List[CutEdge] = []
        total_cost = 0.0

        for idx in sorted(list(final_drop_indices)):
            edge_key = sub_window.edges[idx]
            weight = 0.0
            cost = float(edge_costs[idx])
            total_cost += cost
            cut_edges.append(
                CutEdge(
                    edge=edge_key,
                    mask_weight=round(weight, 4),
                    cost=round(cost, 4),
                    virtual=False,
                    action_type=ActionType.DROP,
                )
            )

        for idx in sorted(list(final_rate_limit_indices)):
            edge_key = sub_window.edges[idx]
            weight = self.rate_limit_factor
            cost = float(edge_costs[idx]) * self.rate_limit_factor
            total_cost += cost
            cut_edges.append(
                CutEdge(
                    edge=edge_key,
                    mask_weight=round(weight, 4),
                    cost=round(cost, 4),
                    virtual=False,
                    action_type=ActionType.RATE_LIMIT,
                )
            )

        # Sort cut edges by intervention severity (DROP before RATE_LIMIT), then cost descending
        cut_edges.sort(key=lambda ce: (0 if ce.action_type == ActionType.DROP else 1, -ce.cost))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        containment_prob = 1.0 if success else 0.0

        msg = (
            f"Containment {'SUCCEEDED' if success else 'FAILED'} in {elapsed_ms:.1f}ms: "
            f"{len(final_drop_indices)} drops, {len(final_rate_limit_indices)} rate-limits, "
            f"cost={total_cost:.3f} (unmasked={unmasked_attacks}, drifted_benign={drifted_benign})"
        )
        logger.info(msg)

        return CutSet(
            edges=cut_edges,
            total_cost=round(total_cost, 4),
            containment_probability=containment_prob,
            status_message=msg,
            metadata={
                "candidate_count": len(candidate_indices),
                "drop_count": len(final_drop_indices),
                "rate_limit_count": len(final_rate_limit_indices),
                "initial_drop_size": len(drop_indices),
                "final_cut_size": len(cut_edges),
                "iterations_used": iters_used,
                "max_remaining_unblocked_score": round(max_rem, 4),
                "failure_reason": fail_reason,
                "unmasked_true_attacks": unmasked_attacks,
                "drifted_benign_edges": drifted_benign,
                "score_drift_count": unmasked_attacks + drifted_benign,
            },
        )

    def _optimize_mask(
        self,
        sub_region: SubgraphRegion,
        detector: Detector,
        candidate_indices: List[int],
        edge_costs: np.ndarray,
    ) -> np.ndarray:
        """Run Adam optimization over continuous mask parameters theta for candidate edges."""
        sub_window = sub_region.window
        num_sub_edges = len(sub_window.edges)

        model = getattr(detector, "model", None)
        if not isinstance(model, nn.Module):
            mask = np.ones(num_sub_edges, dtype=np.float32)
            for idx in candidate_indices:
                if sub_window.edges[idx].id in sub_region.flagged_edge_ids:
                    mask[idx] = 0.0
            return mask

        model.eval()
        pyg_data = sub_window.to_pyg().to(self.device)

        num_candidates = len(candidate_indices)
        theta = nn.Parameter(torch.full((num_candidates,), 1.0, device=self.device))
        optimizer = torch.optim.Adam([theta], lr=self.learning_rate)

        cand_tensor_indices = torch.tensor(candidate_indices, dtype=torch.long, device=self.device)
        costs_tensor = torch.tensor(
            edge_costs[candidate_indices], dtype=torch.float32, device=self.device
        )

        flagged_sub_indices = [
            i for i, e in enumerate(sub_window.edges) if e.id in sub_region.flagged_edge_ids
        ]
        flagged_tensor_indices = torch.tensor(
            flagged_sub_indices, dtype=torch.long, device=self.device
        )

        benign_idx = getattr(detector, "benign_class_idx", 0)

        best_loss = float("inf")
        best_mask = np.ones(num_sub_edges, dtype=np.float32)

        for _ in range(self.max_steps):
            optimizer.zero_grad()

            cand_mask = torch.sigmoid(theta)  # in (0, 1)

            # Continuous probability approximations for DROP and RATE_LIMIT
            # p_drop: M_e < drop_threshold
            p_drop = torch.sigmoid(15.0 * (self.drop_threshold - cand_mask))
            # p_rl: drop_threshold <= M_e < rate_limit_threshold
            p_rl = torch.clamp(
                torch.sigmoid(15.0 * (self.rate_limit_threshold - cand_mask)) - p_drop,
                min=0.0,
                max=1.0,
            )
            # p_allow: M_e >= rate_limit_threshold
            p_allow = torch.clamp(1.0 - p_drop - p_rl, min=0.0, max=1.0)

            # Full mask fed to GNN detector for message-passing evaluation
            full_mask = torch.ones(num_sub_edges, device=self.device)
            # Effective pass-through factor: ALLOW=1.0, RATE_LIMIT=rate_limit_factor, DROP=0.0
            effective_pass = p_allow * 1.0 + p_rl * self.rate_limit_factor + p_drop * 0.0
            full_mask[cand_tensor_indices] = effective_pass

            logits = model(
                pyg_data.x,
                pyg_data.edge_index,
                pyg_data.edge_attr,
                edge_mask=full_mask,
            )

            probs = F.softmax(logits, dim=-1)
            p_benign = probs[:, benign_idx]
            attack_scores = 1.0 - p_benign

            # 1. Flip loss: penalize remaining effective unblocked attack scores on flagged edges
            if len(flagged_sub_indices) > 0:
                flagged_attack = attack_scores[flagged_tensor_indices]
                flagged_pass = full_mask[flagged_tensor_indices]
                effective_unblocked = flagged_pass * flagged_attack
                l_flip = 20.0 * F.binary_cross_entropy(
                    effective_unblocked.clamp(1e-6, 1.0 - 1e-6),
                    torch.zeros_like(effective_unblocked),
                    reduction="sum",
                )
            else:
                l_flip = torch.tensor(0.0, device=self.device)

            # 2. Intervention Cost: DROP charged at cost(e), RATE_LIMIT at rl_factor * cost(e)
            soft_cost_factor = p_drop * 1.0 + p_rl * self.rate_limit_factor
            l_cost = self.lambda1 * torch.sum(costs_tensor * soft_cost_factor)

            # 3. Sparsity penalty: penalize active interventions
            l_sparsity = self.lambda2 * torch.sum(soft_cost_factor)

            # 4. Residual Risk Penalty (Owner Decision 48): penalty for rate-limited attack edges
            if len(flagged_sub_indices) > 0:
                # Find candidate indices corresponding to flagged edges
                flagged_set = set(flagged_sub_indices)
                cand_is_flagged = torch.tensor(
                    [idx in flagged_set for idx in candidate_indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                l_risk = self.lambda_residual_risk * torch.sum(
                    cand_is_flagged * p_rl * self.rate_limit_factor
                )
            else:
                l_risk = torch.tensor(0.0, device=self.device)

            loss = l_flip + l_cost + l_sparsity + l_risk

            loss.backward()
            optimizer.step()

            current_loss = float(loss.item())
            if current_loss < best_loss:
                best_loss = current_loss
                best_mask = cand_mask.detach().cpu().numpy()
                full_best_mask = np.ones(num_sub_edges, dtype=np.float32)
                full_best_mask[candidate_indices] = best_mask

            if len(flagged_sub_indices) > 0 and torch.all(effective_unblocked < self.tau * 0.1):
                full_best_mask = np.ones(num_sub_edges, dtype=np.float32)
                full_best_mask[candidate_indices] = cand_mask.detach().cpu().numpy()
                best_mask = full_best_mask
                break
        else:
            best_mask = full_best_mask

        return best_mask

    def _verify_and_remediate(
        self,
        sub_window: GraphWindow,
        initial_drop_indices: Set[int],
        initial_rate_limit_indices: Set[int],
        candidate_indices: List[int],
        detector: Detector,
    ) -> Tuple[Set[int], Set[int], bool, int, float, str, int, int]:
        """Verify success condition and remediate (Owner Decisions 30 & 48).

        Success condition: Every remaining edge with score >= tau is either cut (DROP)
        or assigned RATE_LIMIT.
        """
        current_drops = set(initial_drop_indices)
        current_rate_limits = set(initial_rate_limit_indices)
        cand_set = set(candidate_indices)
        iterations_used = 0
        max_remaining_score = 0.0
        failure_reason = "None"
        unmasked_attacks = 0
        drifted_benign = 0

        for iteration in range(self.remediation_max_iters + 1):
            iterations_used = iteration
            # Physically remove DROP edges
            remaining_window = self._build_hard_removed_subgraph(sub_window, current_drops)

            if len(remaining_window.edges) == 0:
                return (
                    current_drops,
                    current_rate_limits,
                    True,
                    iterations_used,
                    0.0,
                    "None",
                    0,
                    0,
                )

            # Re-score remaining graph with detector
            det = detector.predict(remaining_window)

            unassigned_violators: List[Tuple[int, float]] = []
            max_remaining_score = 0.0

            for sub_idx, edge in enumerate(sub_window.edges):
                if sub_idx not in current_drops and edge.id in det.edge_scores:
                    score = float(det.edge_scores[edge.id])
                    # If edge is already in RATE_LIMIT, condition is satisfied
                    if sub_idx in current_rate_limits:
                        continue

                    # Otherwise, check if unmitigated edge scores >= tau
                    if score > max_remaining_score:
                        max_remaining_score = score
                    if score >= self.tau:
                        unassigned_violators.append((sub_idx, score))

            if not unassigned_violators:
                # Every remaining edge scoring >= tau is either dropped or rate-limited!
                # Compute score drift metrics
                unmasked_attacks = 0
                drifted_benign = 0
                for sub_idx, edge in enumerate(sub_window.edges):
                    if (
                        sub_idx not in cand_set
                        and sub_idx not in current_drops
                        and edge.id in det.edge_scores
                    ):
                        if float(det.edge_scores[edge.id]) >= self.tau:
                            is_true_attack = (
                                sub_window.labels is not None and sub_window.labels[sub_idx] > 0
                            )
                            if is_true_attack:
                                unmasked_attacks += 1
                            else:
                                drifted_benign += 1

                return (
                    current_drops,
                    current_rate_limits,
                    True,
                    iterations_used,
                    max_remaining_score,
                    "None",
                    unmasked_attacks,
                    drifted_benign,
                )

            if iteration == self.remediation_max_iters:
                # Assign all remaining violators to RATE_LIMIT to satisfy condition
                for viol_idx, _ in unassigned_violators:
                    current_rate_limits.add(viol_idx)
                return (
                    current_drops,
                    current_rate_limits,
                    True,
                    iterations_used,
                    max_remaining_score,
                    "None",
                    unmasked_attacks,
                    drifted_benign,
                )

            # Remediate highest scoring unassigned violator by assigning RATE_LIMIT
            unassigned_violators.sort(key=lambda x: x[1], reverse=True)
            viol_idx_to_remediate = unassigned_violators[0][0]
            current_rate_limits.add(viol_idx_to_remediate)

        return (
            current_drops,
            current_rate_limits,
            True,
            iterations_used,
            max_remaining_score,
            failure_reason,
            unmasked_attacks,
            drifted_benign,
        )

    def _build_hard_removed_subgraph(
        self, sub_window: GraphWindow, drop_indices: Set[int]
    ) -> GraphWindow:
        """Construct a new GraphWindow with DROP edges physically deleted."""
        kept_edges: List[EdgeKey] = []
        kept_attr: List[List[float]] = []
        kept_labels: Optional[List[int]] = [] if sub_window.labels is not None else None

        for i, edge in enumerate(sub_window.edges):
            if i not in drop_indices:
                kept_edges.append(edge)
                kept_attr.append(sub_window.edge_attr[i])
                if kept_labels is not None and sub_window.labels is not None:
                    kept_labels.append(sub_window.labels[i])

        if not kept_edges:
            return GraphWindow(
                window_id=f"{sub_window.window_id or 'win'}_hard_cut",
                t_start=sub_window.t_start,
                t_end=sub_window.t_end,
                node_ids=[],
                edge_index=[],
                edge_attr=[],
                edges=[],
                labels=None,
            )

        unique_nodes = sorted(list({e.src_ip for e in kept_edges} | {e.dst_ip for e in kept_edges}))
        node_map = {nid: i for i, nid in enumerate(unique_nodes)}

        edge_index: List[List[int]] = [[node_map[e.src_ip], node_map[e.dst_ip]] for e in kept_edges]

        return GraphWindow(
            window_id=f"{sub_window.window_id or 'win'}_hard_cut",
            t_start=sub_window.t_start,
            t_end=sub_window.t_end,
            node_ids=unique_nodes,
            edge_index=edge_index,
            edge_attr=kept_attr,
            edges=kept_edges,
            labels=kept_labels,
        )


class GreedyCutSolver(CutSolver):
    """Greedy cut baseline: iteratively cuts edges with highest attack_score / cost utility."""

    def __init__(
        self,
        tau_low: float = 0.30,
        tau: float = 0.50,
        k_hops: int = 2,
        max_cuts: Optional[int] = None,
        lambda_cost: float = 0.0,
    ) -> None:
        self.tau_low = tau_low
        self.tau = tau
        self.k_hops = k_hops
        self.max_cuts = max_cuts
        self.lambda_cost = lambda_cost

    def solve(
        self,
        incident: Incident,
        window: GraphWindow,
        detector: Detector,
        cost_fn: Callable[[EdgeKey, GraphWindow], float],
    ) -> CutSet:
        start_time = time.perf_counter()

        flagged_edge_ids: List[str] = [e.id for e in incident.target_edges]
        if not flagged_edge_ids:
            det = detector.predict(window)
            flagged_edge_ids = det.flagged_edges
        else:
            det = detector.predict(window)

        if not flagged_edge_ids or not window.edges:
            return CutSet(
                edges=[],
                total_cost=0.0,
                containment_probability=1.0,
                status_message="No flagged edges",
                metadata={
                    "candidate_count": 0,
                    "initial_cut_size": 0,
                    "final_cut_size": 0,
                    "iterations_used": 0,
                    "max_remaining_unblocked_score": 0.0,
                    "failure_reason": "None",
                },
            )

        sub_region = extract_khop_subgraph(
            window=window,
            flagged_edge_ids=flagged_edge_ids,
            detection=det,
            k_hops=self.k_hops,
            tau_low=self.tau_low,
        )

        sub_window = sub_region.window
        candidate_indices = sub_region.candidate_edge_indices

        if not candidate_indices:
            return CutSet(
                edges=[],
                total_cost=0.0,
                containment_probability=0.0,
                status_message="No candidates",
                metadata={
                    "candidate_count": 0,
                    "initial_cut_size": 0,
                    "final_cut_size": 0,
                    "iterations_used": 0,
                    "max_remaining_unblocked_score": 0.0,
                    "failure_reason": "No candidate edges in subgraph",
                },
            )

        # Uncapped greedy: allow cutting up to all candidates unless explicitly capped
        effective_max_cuts = self.max_cuts if self.max_cuts is not None else len(candidate_indices)

        # Rank candidates by utility = score / (1.0 + lambda_cost * cost)
        candidate_scores = []
        for idx in candidate_indices:
            edge = sub_window.edges[idx]
            score = float(det.edge_scores.get(edge.id, 0.0))
            cost = float(cost_fn(edge, window))
            utility = score / (1.0 + self.lambda_cost * cost)
            candidate_scores.append((idx, score, cost, utility))

        candidate_scores.sort(key=lambda x: x[3], reverse=True)

        current_cut: Set[int] = set()
        success = False
        iters_used = 0
        max_rem = 0.0
        fail_reason = "None"

        # Check initial state before cuts
        remaining = self._build_hard_removed(sub_window, current_cut)
        if len(remaining.edges) == 0:
            success = True
            max_rem = 0.0
        else:
            rem_det = detector.predict(remaining)
            current_violators = [
                idx
                for idx in candidate_indices
                if idx not in current_cut
                and sub_window.edges[idx].id in rem_det.edge_scores
                and float(rem_det.edge_scores[sub_window.edges[idx].id]) >= self.tau
            ]
            cand_scores = [
                float(rem_det.edge_scores[sub_window.edges[idx].id])
                for idx in candidate_indices
                if idx not in current_cut and sub_window.edges[idx].id in rem_det.edge_scores
            ]
            max_rem = max(cand_scores) if cand_scores else 0.0
            if not current_violators:
                success = True

        # Iterative remediation loop (cutting all active violators per iteration)
        if not success:
            for iter_step in range(10):
                iters_used = iter_step + 1

                # Rank current violators by utility: score / (1.0 + lambda_cost * cost)
                violator_scores = []
                for idx in current_violators:
                    edge = sub_window.edges[idx]
                    score = float(rem_det.edge_scores.get(edge.id, 0.0))
                    cost = float(cost_fn(edge, window))
                    utility = score / (1.0 + self.lambda_cost * cost)
                    violator_scores.append((idx, score, cost, utility))

                violator_scores.sort(key=lambda x: x[3], reverse=True)

                eligible_to_cut = []
                for idx, score, cost, utility in violator_scores:
                    if self.lambda_cost > 0.0 and utility < (self.tau * 0.1):
                        fail_reason = (
                            f"Stopped early: utility {utility:.4f} below cost penalty threshold"
                        )
                        break
                    eligible_to_cut.append(idx)

                if not eligible_to_cut:
                    if fail_reason == "None":
                        fail_reason = "No eligible violators to cut"
                    break

                added = 0
                for idx in eligible_to_cut:
                    if len(current_cut) >= effective_max_cuts:
                        fail_reason = (
                            f"Exhausted max_cuts={effective_max_cuts} "
                            f"(max unblocked: {max_rem:.4f})"
                        )
                        break
                    current_cut.add(idx)
                    added += 1

                if added == 0:
                    break

                # Re-evaluate remaining subgraph
                remaining = self._build_hard_removed(sub_window, current_cut)
                if len(remaining.edges) == 0:
                    success = True
                    max_rem = 0.0
                    fail_reason = "None"
                    break

                rem_det = detector.predict(remaining)
                cand_scores = [
                    float(rem_det.edge_scores[sub_window.edges[idx].id])
                    for idx in candidate_indices
                    if idx not in current_cut and sub_window.edges[idx].id in rem_det.edge_scores
                ]
                max_rem = max(cand_scores) if cand_scores else 0.0

                current_violators = [
                    idx
                    for idx in candidate_indices
                    if idx not in current_cut
                    and sub_window.edges[idx].id in rem_det.edge_scores
                    and float(rem_det.edge_scores[sub_window.edges[idx].id]) >= self.tau
                ]
                if not current_violators:
                    success = True
                    fail_reason = "None"
                    break
            else:
                if not success and fail_reason == "None":
                    fail_reason = f"Exhausted iterations (max unblocked: {max_rem:.4f})"

        # Score drift accounting: separate unmasked true attacks from drifted benign edges
        rem_final = self._build_hard_removed(sub_window, current_cut)
        rem_det_final = detector.predict(rem_final) if len(rem_final.edges) > 0 else None
        unmasked_attacks = 0
        drifted_benign = 0
        if rem_det_final:
            for idx, e in enumerate(sub_window.edges):
                if (
                    idx not in candidate_indices
                    and idx not in current_cut
                    and e.id in rem_det_final.edge_scores
                ):
                    if float(rem_det_final.edge_scores[e.id]) >= self.tau:
                        is_true_attack = (
                            sub_window.labels is not None and sub_window.labels[idx] > 0
                        )
                        if is_true_attack:
                            unmasked_attacks += 1
                        else:
                            drifted_benign += 1

        cut_edges = []
        total_cost = 0.0
        for idx in sorted(list(current_cut)):
            edge = sub_window.edges[idx]
            c = float(cost_fn(edge, window))
            total_cost += c
            cut_edges.append(
                CutEdge(
                    edge=edge,
                    mask_weight=0.0,
                    cost=round(c, 4),
                    virtual=False,
                    action_type=ActionType.DROP,
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        msg = f"Greedy cut {len(cut_edges)} edges in {elapsed_ms:.1f}ms (success={success})"
        return CutSet(
            edges=cut_edges,
            total_cost=round(total_cost, 4),
            containment_probability=1.0 if success else 0.0,
            status_message=msg,
            metadata={
                "candidate_count": len(candidate_indices),
                "initial_cut_size": 0,
                "final_cut_size": len(cut_edges),
                "iterations_used": iters_used,
                "max_remaining_unblocked_score": round(max_rem, 4),
                "failure_reason": fail_reason,
                "unmasked_true_attacks": unmasked_attacks,
                "drifted_benign_edges": drifted_benign,
                "score_drift_count": unmasked_attacks + drifted_benign,
            },
        )

    def _build_hard_removed(self, sub_window: GraphWindow, cut_indices: Set[int]) -> GraphWindow:
        kept_edges = [e for i, e in enumerate(sub_window.edges) if i not in cut_indices]
        kept_attr = [a for i, a in enumerate(sub_window.edge_attr) if i not in cut_indices]
        if not kept_edges:
            return GraphWindow(
                window_id=f"{sub_window.window_id or 'win'}_greedy_cut",
                t_start=sub_window.t_start,
                t_end=sub_window.t_end,
                node_ids=[],
                edge_index=[],
                edge_attr=[],
                edges=[],
                labels=None,
            )

        unique_nodes = sorted(list({e.src_ip for e in kept_edges} | {e.dst_ip for e in kept_edges}))
        node_map = {nid: i for i, nid in enumerate(unique_nodes)}

        edge_index = [[node_map[e.src_ip], node_map[e.dst_ip]] for e in kept_edges]

        return GraphWindow(
            window_id=f"{sub_window.window_id or 'win'}_greedy_cut",
            t_start=sub_window.t_start,
            t_end=sub_window.t_end,
            node_ids=unique_nodes,
            edge_index=edge_index,
            edge_attr=kept_attr,
            edges=kept_edges,
            labels=None,
        )

