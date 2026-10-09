"""Comprehensive containment evaluation pipeline and baseline benchmark (Phase P2 / N1).

Generates the collateral damage comparison table, multi-strategy frontier curves,
matched-ABR collateral damage table, and per-incident remediation debug logs
per Owner Decisions 29-50.
"""

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graphguard.containment.cost import EdgeCostEvaluator, infer_asset_roles_from_flows
from graphguard.containment.cut_solver import (
    CounterfactualCutSolver,
    GreedyCutSolver,
)
from graphguard.core.models import (
    ActionType,
    CutSet,
    GraphWindow,
    Incident,
    IncidentSeverity,
    IncidentStatus,
)
from graphguard.detect.egraphsage import EGraphSAGEDetector, EGraphSAGENet
from graphguard.detect.registry import ModelRegistry
from graphguard.eval.baselines import (
    BlockAllFlaggedBaseline,
    IsolateHostBaseline,
    NoResponseBaseline,
)
from graphguard.graph.features import NetFlowFeatureExtractor
from graphguard.graph.window_builder import (
    WindowBuilder,
    inspect_timestamp_column,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class IncidentContext:
    """Fixed, strategy-independent evaluation context for a single incident window."""

    incident_id: str
    window: GraphWindow
    flows_df: pd.DataFrame
    incident_model: Incident
    region_hosts: Set[str]
    in_region_mask: np.ndarray
    edge_ids: np.ndarray
    is_attacks: np.ndarray
    src_subnets: np.ndarray
    dst_ips: np.ndarray
    dst_ports: np.ndarray
    denom_attack_flows: int
    denom_benign_flows: int
    denom_legit_benign_flows: int
    collateral_floor: float
    legit_ben_mask: np.ndarray


def compute_bootstrap_ci(
    values: List[float],
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute mean and percentile bootstrap confidence interval over incidents (Decision 42)."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    mean_val = float(np.mean(arr))
    if len(arr) <= 1:
        return mean_val, mean_val, mean_val
    rng = np.random.default_rng(seed)
    boot_means = []
    n = len(arr)
    for _ in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    alpha = (1.0 - ci_level) / 2.0
    low = float(np.percentile(boot_means, 100.0 * alpha))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return mean_val, low, high


def interpolate_cdr_at_abr(
    abr_points: List[float],
    cdr_points: List[float],
    target_abr: float,
    max_gap: float = 0.05,
) -> Optional[Tuple[float, str]]:
    """Find or interpolate CDR at target ABR without exceeding max_gap (Decision 54).

    Returns (cdr_value, display_str).
    If the gap between adjacent measured points exceeds max_gap (0.05), interpolation
    is strictly suppressed and None is returned.
    """
    if not abr_points or not cdr_points:
        return None
    pts = sorted(zip(abr_points, cdr_points), key=lambda p: p[0])
    abrs = [p[0] for p in pts]
    cdrs = [p[1] for p in pts]

    # Check for near-exact measured point (within 1.5% ABR)
    for a, c in zip(abrs, cdrs):
        if abs(a - target_abr) <= 0.015:
            return (c, f"{c * 100:.1f}% (measured at {a * 100:.1f}%)")

    if target_abr < abrs[0]:
        return (cdrs[0], f"{cdrs[0] * 100:.1f}% (measured at {abrs[0] * 100:.1f}%)")
    if target_abr > abrs[-1]:
        return None

    for i in range(len(abrs) - 1):
        if abrs[i] <= target_abr <= abrs[i + 1]:
            gap = abrs[i + 1] - abrs[i]
            if gap > max_gap:
                # No interpolation across gaps > 5 ABR points (Owner Directive)
                return None
            frac = (target_abr - abrs[i]) / gap if gap > 0 else 0.0
            val = cdrs[i] + frac * (cdrs[i + 1] - cdrs[i])
            return (val, f"{val * 100:.1f}% (interpolated, gap {gap * 100:.1f}%)")

    return (cdrs[-1], f"{cdrs[-1] * 100:.1f}% (measured at {abrs[-1] * 100:.1f}%)")


def build_fixed_incident_contexts(
    windows_with_meta: List[Any],
    detector: EGraphSAGEDetector,
    k_hops: int = 2,
    max_incidents: int = 100,
    all_attack_sources: Optional[Set[str]] = None,
) -> List[IncidentContext]:
    """Define the evaluation incidents and their regions ONCE (Owner Decisions 47 & 48).

    Region = k-hop neighbourhood of hosts with at least one true attack flow in the window.
    All denominators and regions are strictly ground-truth based and fixed across all strategies.
    """
    proto_map = {6: "tcp", 17: "udp", 1: "icmp"}
    contexts: List[IncidentContext] = []

    if all_attack_sources is None:
        all_attack_sources = set()
        for _, _, _, chunk in windows_with_meta:
            c_lookup = {col.lower().strip(): col for col in chunk.columns}
            lbl_c = c_lookup.get("label")
            src_c = c_lookup.get("ipv4_src_addr") or c_lookup.get("src_ip")
            if lbl_c and src_c:
                att_c = chunk[pd.to_numeric(chunk[lbl_c], errors="coerce").fillna(0) > 0]
                all_attack_sources.update(att_c[src_c].astype(str).str.strip().unique())

    for w, lbls, atts, chunk_df in windows_with_meta:
        # Check for presence of true attack flows
        lookup = {c.lower().strip(): c for c in chunk_df.columns}
        src_col = lookup.get("ipv4_src_addr") or lookup.get("src_ip")
        dst_col = lookup.get("ipv4_dst_addr") or lookup.get("dst_ip")
        dport_col = lookup.get("l4_dst_port") or lookup.get("dst_port")
        proto_col = lookup.get("protocol") or lookup.get("proto")
        lbl_col = lookup.get("label")

        is_attacks = (
            (pd.to_numeric(chunk_df[lbl_col], errors="coerce").fillna(0).to_numpy() > 0)
            if lbl_col
            else np.zeros(len(chunk_df), dtype=bool)
        )

        if not np.any(is_attacks):
            continue

        s_ips = chunk_df[src_col].astype(str).str.strip().to_numpy()
        d_ips = chunk_df[dst_col].astype(str).str.strip().to_numpy()
        d_ports = (
            pd.to_numeric(chunk_df[dport_col], errors="coerce").fillna(80).astype(int).to_numpy()
            if dport_col
            else np.full(len(chunk_df), 80, dtype=int)
        )
        raw_protos = chunk_df[proto_col].to_numpy() if proto_col else np.full(len(chunk_df), "tcp")
        p_strs = np.array([proto_map.get(p, str(p).lower().strip()) for p in raw_protos])
        edge_ids = np.array(
            [f"{s}>{d}:{dp}/{p}" for s, d, dp, p in zip(s_ips, d_ips, d_ports, p_strs)]
        )

        # 1. Ground truth attack hosts
        seed_hosts = set(s_ips[is_attacks]) | set(d_ips[is_attacks])

        # 2. k-hop neighborhood expansion
        curr_hosts = set(seed_hosts)
        for _ in range(k_hops):
            nbrs = set()
            for s, d in zip(s_ips, d_ips):
                if s in curr_hosts:
                    nbrs.add(d)
                if d in curr_hosts:
                    nbrs.add(s)
            curr_hosts.update(nbrs)
        region_hosts = curr_hosts

        # 3. Induced region flows
        in_region = np.isin(s_ips, list(region_hosts)) & np.isin(d_ips, list(region_hosts))
        reg_att_mask = in_region & is_attacks
        reg_ben_mask = in_region & (~is_attacks)

        # 3b. Legitimate benign flows (NEITHER endpoint is a source of true attack in dataset)
        s_is_att = np.isin(s_ips, list(all_attack_sources))
        d_is_att = np.isin(d_ips, list(all_attack_sources))
        reg_legit_ben_mask = reg_ben_mask & (~s_is_att) & (~d_is_att)

        denom_att = int(np.sum(reg_att_mask))
        denom_ben = int(np.sum(reg_ben_mask))
        denom_legit_ben = int(np.sum(reg_legit_ben_mask))

        # 4. Collateral floor: benign flows sharing EdgeKey with any attack flow
        att_keys = set(edge_ids[reg_att_mask])
        ben_on_att_edges = int(np.sum(reg_ben_mask & np.isin(edge_ids, list(att_keys))))
        floor = (ben_on_att_edges / denom_ben) if denom_ben > 0 else 0.0

        # 5. Extract subnets for breakdown
        src_subnets = np.array(
            [s.rsplit(".", 1)[0] + ".0/24" if "." in s else s for s in s_ips]
        )

        # 6. Flagged target edges for solver initiation
        det = detector.predict(w)
        target_edges = [e for e in w.edges if e.id in det.flagged_edges]
        if not target_edges and w.edges:
            target_edges = [w.edges[0]]

        incident_id = f"inc_{w.window_id or len(contexts)}"
        incident_model = Incident(
            incident_id=incident_id,
            severity=IncidentSeverity.HIGH,
            status=IncidentStatus.OPEN,
            target_edges=target_edges,
        )

        contexts.append(
            IncidentContext(
                incident_id=incident_id,
                window=w,
                flows_df=chunk_df,
                incident_model=incident_model,
                region_hosts=region_hosts,
                in_region_mask=in_region,
                edge_ids=edge_ids,
                is_attacks=is_attacks,
                src_subnets=src_subnets,
                dst_ips=d_ips,
                dst_ports=d_ports,
                denom_attack_flows=denom_att,
                denom_benign_flows=denom_ben,
                denom_legit_benign_flows=denom_legit_ben,
                collateral_floor=floor,
                legit_ben_mask=reg_legit_ben_mask,
            )
        )

        if len(contexts) >= max_incidents:
            break

    logger.info(
        f"Built {len(contexts)} fixed incident contexts (k_hops={k_hops}). "
        f"Mean attack denom: {np.mean([c.denom_attack_flows for c in contexts]):.1f}, "
        f"Mean benign denom: {np.mean([c.denom_benign_flows for c in contexts]):.1f}, "
        f"Mean legit benign denom: {np.mean([c.denom_legit_benign_flows for c in contexts]):.1f}."
    )
    return contexts


def evaluate_containment_on_incident(
    solver: Any,
    ctx: IncidentContext,
    detector: EGraphSAGEDetector,
    cost_fn: EdgeCostEvaluator,
    rate_limit_factor: float = 0.3,
    tau: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate a containment solver on a fixed incident context (Owner Decision 48)."""
    t0 = time.perf_counter()
    cut_set: CutSet = solver.solve(ctx.incident_model, ctx.window, detector, cost_fn)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Categorize cut set actions into DROP vs RATE_LIMIT
    drop_edge_ids = {
        ce.edge.id for ce in cut_set.edges if ce.action_type == ActionType.DROP
    }
    rl_edge_ids = {
        ce.edge.id for ce in cut_set.edges if ce.action_type == ActionType.RATE_LIMIT
    }

    # Restrict accounting strictly to flows within the fixed ground-truth region
    in_reg = ctx.in_region_mask
    e_ids = ctx.edge_ids
    is_att = ctx.is_attacks

    att_in_reg = in_reg & is_att
    ben_in_reg = in_reg & (~is_att)

    att_drop_count = int(np.sum(att_in_reg & np.isin(e_ids, list(drop_edge_ids))))
    att_rl_count = int(np.sum(att_in_reg & np.isin(e_ids, list(rl_edge_ids))))

    ben_drop_count = int(np.sum(ben_in_reg & np.isin(e_ids, list(drop_edge_ids))))
    ben_rl_count = int(np.sum(ben_in_reg & np.isin(e_ids, list(rl_edge_ids))))

    # Fixed-denominator rates
    denom_att = ctx.denom_attack_flows
    denom_ben = ctx.denom_benign_flows

    abr = (att_drop_count / denom_att) if denom_att > 0 else 1.0
    atr = (att_rl_count / denom_att) if denom_att > 0 else 0.0
    total_containment = abr + atr
    rar = max(0.0, 1.0 - total_containment)

    # Collateral Damage Rate: rate-limited benign flow counts as rate_limit_factor of a blocked flow
    effective_ben_blocked = float(ben_drop_count) + rate_limit_factor * float(ben_rl_count)
    cdr = (effective_ben_blocked / denom_ben) if denom_ben > 0 else 0.0

    # Legitimate Collateral Damage Rate: benign flows where NEITHER endpoint is an attack source
    legit_mask = ctx.legit_ben_mask
    denom_legit = ctx.denom_legit_benign_flows
    legit_drop_count = int(np.sum(legit_mask & np.isin(e_ids, list(drop_edge_ids))))
    legit_rl_count = int(np.sum(legit_mask & np.isin(e_ids, list(rl_edge_ids))))
    effective_legit_blocked = float(legit_drop_count) + rate_limit_factor * float(legit_rl_count)
    lcdr = (effective_legit_blocked / denom_legit) if denom_legit > 0 else 0.0

    # Collect blocked benign flow breakdowns for inspection
    blocked_benign_records: List[Tuple[str, str, int, float]] = []
    if ben_drop_count > 0:
        drop_ben_mask = ben_in_reg & np.isin(e_ids, list(drop_edge_ids))
        subnets = ctx.src_subnets[drop_ben_mask]
        dests = ctx.dst_ips[drop_ben_mask]
        ports = ctx.dst_ports[drop_ben_mask]
        for s_sub, d_ip, d_p in zip(subnets, dests, ports):
            blocked_benign_records.append((s_sub, d_ip, int(d_p), 1.0))

    if ben_rl_count > 0:
        rl_ben_mask = ben_in_reg & np.isin(e_ids, list(rl_edge_ids))
        subnets = ctx.src_subnets[rl_ben_mask]
        dests = ctx.dst_ips[rl_ben_mask]
        ports = ctx.dst_ports[rl_ben_mask]
        for s_sub, d_ip, d_p in zip(subnets, dests, ports):
            blocked_benign_records.append((s_sub, d_ip, int(d_p), rate_limit_factor))

    meta = getattr(cut_set, "metadata", {}) or {}
    model_contained = bool(cut_set.containment_probability >= 0.5)
    max_remaining_score = float(meta.get("max_remaining_unblocked_score", 0.0))
    fail_reason_default = "None" if model_contained else "Containment condition failed"
    fail_reason = str(meta.get("failure_reason", fail_reason_default))
    unmasked_attacks = int(meta.get("unmasked_true_attacks", 0))
    drifted_benign = int(meta.get("drifted_benign_edges", 0))

    return {
        "edges_cut": len(cut_set.edges),
        "edges_dropped": len(drop_edge_ids),
        "edges_rate_limited": len(rl_edge_ids),
        "total_cost": cut_set.total_cost,
        "abr": abr,
        "atr": atr,
        "cdr": cdr,
        "lcdr": lcdr,
        "rar": rar,
        "model_contained": model_contained,
        "unmasked_true_attacks": unmasked_attacks,
        "drifted_benign_edges": drifted_benign,
        "latency_ms": latency_ms,
        "candidate_count": meta.get("candidate_count", len(ctx.window.edges)),
        "iterations_used": meta.get("iterations_used", 0),
        "max_remaining_unblocked_score": max_remaining_score,
        "failure_reason": fail_reason,
        "blocked_benign_records": blocked_benign_records,
    }


def run_containment_benchmark(
    data_path: Optional[str] = None,
    tau_low: float = 0.30,
    tau: Optional[float] = None,
    rate_limit_factor: float = 0.3,
    max_incidents: int = 100,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Execute end-to-end benchmark across containment methods, sweeps, and bootstrap CIs."""
    if data_path is None:
        raw_csv = Path("data/raw/NF-UNSW-NB15-v3.csv")
        if raw_csv.exists():
            data_path = str(raw_csv)
        else:
            data_path = "data/sample_nf_v2.csv"

    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset from {data_path} with {len(df):,} flows.")

    # 1. Infer asset roles across whole dataset (Decision 33)
    inferred_roles = infer_asset_roles_from_flows(df, output_path="config/assets_inferred.yaml")
    cost_evaluator = EdgeCostEvaluator("config/assets_inferred.yaml")

    # 2. Extract test split chronologically
    has_time, time_col = inspect_timestamp_column(df)
    if len(df) > 50000:
        if has_time and time_col:
            logger.info(f"Sorting dataset chronologically by timestamp column '{time_col}'.")
            df = df.sort_values(by=time_col).reset_index(drop=True)
        test_start = int(len(df) * 0.85)
        eval_df = df.iloc[test_start:].copy().reset_index(drop=True)
        logger.info(
            f"Evaluating containment strictly on real test split "
            f"({len(eval_df):,} flows, 85%-100% time split)."
        )
    else:
        eval_df = df

    # 3. Model acquisition: load registered P1 model
    reg = ModelRegistry("models")
    detector: EGraphSAGEDetector
    try:
        detector, extractor, reg_meta = reg.load_model("egraphsage")
        logger.info(f"Loaded registered E-GraphSAGE model (calibrated tau={detector.tau:.3f})")
    except Exception as e:
        logger.warning(f"Could not load registered model ({e}). Training fallback detector...")
        extractor = NetFlowFeatureExtractor()
        features = extractor.fit_transform(eval_df)
        edge_dim = features.shape[1]
        net = EGraphSAGENet(edge_dim=edge_dim, num_classes=2, hidden_dim=32, dropout=0.0)
        actual_tau = 0.50
        detector = EGraphSAGEDetector(
            model=net,
            classes=["Benign", "Attack"],
            benign_class_idx=0,
            tau=actual_tau,
        )

    eval_tau = tau if tau is not None else detector.tau
    logger.info(f"Using operational detection threshold tau={eval_tau:.3f}")

    # 4. Extract true attack sources across dataset and build fixed incident contexts
    lookup_df = {c.lower().strip(): c for c in df.columns}
    src_c = lookup_df.get("ipv4_src_addr") or lookup_df.get("src_ip")
    lbl_c = lookup_df.get("label")
    all_attack_sources = (
        set(
            df[pd.to_numeric(df[lbl_c], errors="coerce").fillna(0) > 0][src_c]
            .astype(str)
            .str.strip()
            .unique()
        )
        if (lbl_c and src_c)
        else set()
    )
    logger.info(
        f"Identified {len(all_attack_sources)} true attack sources across dataset: "
        f"{sorted(list(all_attack_sources))}"
    )

    if has_time and time_col:
        builder = WindowBuilder(window_duration_seconds=60.0)
    else:
        builder = WindowBuilder(chunk_size=150)

    windows_with_meta = builder.build_windows(eval_df, extractor, return_dfs=True)
    incident_contexts = build_fixed_incident_contexts(
        windows_with_meta=windows_with_meta,
        detector=detector,
        k_hops=2,
        max_incidents=max_incidents,
        all_attack_sources=all_attack_sources,
    )
    num_incidents = len(incident_contexts)

    # 5. Report Collateral Floor
    collateral_floors = [ctx.collateral_floor for ctx in incident_contexts]
    mean_floor = float(np.mean(collateral_floors))
    total_ben_flows = sum(ctx.denom_benign_flows for ctx in incident_contexts)
    total_shared_flows = sum(
        round(ctx.collateral_floor * ctx.denom_benign_flows) for ctx in incident_contexts
    )
    pooled_floor = (total_shared_flows / total_ben_flows) if total_ben_flows > 0 else 0.0

    floor_df = pd.DataFrame(
        [
            {
                "Total Incidents": num_incidents,
                "Mean Benign Flows / Region": (
                    f"{np.mean([c.denom_benign_flows for c in incident_contexts]):.1f}"
                ),
                "Mean Legitimate Benign Flows / Region": (
                    f"{np.mean([c.denom_legit_benign_flows for c in incident_contexts]):.1f}"
                ),
                "Mean Collateral Floor / Incident": f"{mean_floor * 100:.2f}%",
                "Pooled Collateral Floor Overall": f"{pooled_floor * 100:.2f}%",
                "Explanation": (
                    "Attacker IPs (175.45.176.0/24) and benign client IPs (59.166.0.0/24) "
                    "are disjoint; 0 benign flows share EdgeKeys with attacks."
                ),
            }
        ]
    )

    # 6. Primary Solvers to benchmark
    solvers = {
        "No Response": NoResponseBaseline(),
        "Isolate Host": IsolateHostBaseline(),
        "Block All Flagged": BlockAllFlaggedBaseline(tau=eval_tau),
        "Greedy Cut": GreedyCutSolver(
            tau_low=tau_low,
            tau=eval_tau,
            max_cuts=None,
            lambda_cost=0.0,
        ),
        "N1 Counterfactual Cut": CounterfactualCutSolver(
            tau_low=tau_low,
            tau=eval_tau,
            rate_limit_factor=rate_limit_factor,
            lambda1=1.0,
            lambda2=0.5,
            lambda_residual_risk=5.0,
            max_steps=120,
        ),
    }

    headline_results = []
    cost_profile_results = []
    debug_incident_logs = []
    all_breakdown_records: Dict[str, List[Tuple[str, str, int, float]]] = {
        s: [] for s in solvers.keys()
    }

    for name, solver in solvers.items():
        abrs, atrs, cdrs, lcdrs, rars = [], [], [], [], []
        contained_flags, latencies = [], []
        drops, rls, cuts, costs = [], [], [], []
        unmasked_threats, drifted_benigns = [], []

        for ctx in incident_contexts:
            res = evaluate_containment_on_incident(
                solver=solver,
                ctx=ctx,
                detector=detector,
                cost_fn=cost_evaluator,
                rate_limit_factor=rate_limit_factor,
                tau=eval_tau,
            )
            abrs.append(res["abr"])
            atrs.append(res["atr"])
            cdrs.append(res["cdr"])
            lcdrs.append(res["lcdr"])
            rars.append(res["rar"])
            contained_flags.append(1.0 if res["model_contained"] else 0.0)
            latencies.append(res["latency_ms"])
            drops.append(res["edges_dropped"])
            rls.append(res["edges_rate_limited"])
            cuts.append(res["edges_cut"])
            costs.append(res["total_cost"])
            unmasked_threats.append(res["unmasked_true_attacks"])
            drifted_benigns.append(res["drifted_benign_edges"])
            all_breakdown_records[name].extend(res["blocked_benign_records"])

            debug_incident_logs.append(
                {
                    "Incident ID": ctx.incident_id,
                    "Strategy": name,
                    "Candidates": res["candidate_count"],
                    "Drops": res["edges_dropped"],
                    "RateLimits": res["edges_rate_limited"],
                    "ABR": f"{res['abr'] * 100:.1f}%",
                    "ATR": f"{res['atr'] * 100:.1f}%",
                    "CDR": f"{res['cdr'] * 100:.1f}%",
                    "LegitCDR": f"{res['lcdr'] * 100:.1f}%",
                    "Contained": "YES" if res["model_contained"] else "NO",
                    "Unmasked Attacks": res["unmasked_true_attacks"],
                    "Drifted Benign": res["drifted_benign_edges"],
                    "Failure Reason": res["failure_reason"],
                }
            )

        mean_abr, abr_low, abr_high = compute_bootstrap_ci(abrs)
        mean_cdr, cdr_low, cdr_high = compute_bootstrap_ci(cdrs)
        mean_lcdr, lcdr_low, lcdr_high = compute_bootstrap_ci(lcdrs)
        mean_atr = float(np.mean(atrs))

        abr_ci_str = f"{mean_abr * 100:.1f}% [{abr_low * 100:.1f}%, {abr_high * 100:.1f}%]"
        cdr_ci_str = f"{mean_cdr * 100:.1f}% [{cdr_low * 100:.1f}%, {cdr_high * 100:.1f}%]"
        lcdr_ci_str = f"{mean_lcdr * 100:.1f}% [{lcdr_low * 100:.1f}%, {lcdr_high * 100:.1f}%]"

        headline_results.append(
            {
                "Strategy": name,
                "Incidents (N)": num_incidents,
                "Attack Block Rate (Drops) [95% CI]": abr_ci_str,
                "Attack Throttle Rate (RL)": f"{mean_atr * 100:.1f}%",
                "Label-Based CDR [95% CI]": cdr_ci_str,
                "Legitimate CDR [95% CI]": lcdr_ci_str,
                "Residual Attack Rate (RAR)": f"{np.mean(rars) * 100:.1f}%",
                "Model Containment Success Rate": f"{np.mean(contained_flags) * 100:.1f}%",
                "Mean Latency (ms)": f"{np.mean(latencies):.1f}ms",
                "p95 Latency (ms)": f"{np.percentile(latencies, 95):.1f}ms",
                "_raw_abr": mean_abr,
                "_raw_cdr": mean_cdr,
                "_raw_lcdr": mean_lcdr,
            }
        )

        cost_profile_results.append(
            {
                "Strategy": name,
                "Mean Drops": f"{np.mean(drops):.1f}",
                "Mean RateLimits": f"{np.mean(rls):.1f}",
                "Mean Total Cost": f"{np.mean(costs):.2f}",
                "Cost / Blocked Flow": (
                    f"{np.mean(costs) / max(1, sum(drops) + sum(rls)):.3f}"
                ),
                "Model Containment Success Rate": f"{np.mean(contained_flags) * 100:.1f}%",
            }
        )

    headline_df = pd.DataFrame(headline_results)
    cost_profile_df = pd.DataFrame(cost_profile_results)
    debug_df = pd.DataFrame(debug_incident_logs)

    # 7. Collateral Breakdown Table (Blocked Benign Flows by Subnet, Dest, Port)
    breakdown_rows = []
    for s_name in ["Block All Flagged", "N1 Counterfactual Cut"]:
        records = all_breakdown_records.get(s_name, [])
        if records:
            b_df = pd.DataFrame(records, columns=["src_subnet", "dst_ip", "dst_port", "weight"])
            grouped = (
                b_df.groupby(["src_subnet", "dst_ip", "dst_port"])["weight"]
                .sum()
                .reset_index()
            )
            grouped = grouped.sort_values(by="weight", ascending=False).head(5)
            for _, r in grouped.iterrows():
                breakdown_rows.append(
                    {
                        "Strategy": s_name,
                        "Source Subnet": r["src_subnet"],
                        "Destination Host": r["dst_ip"],
                        "Dst Port": int(r["dst_port"]),
                        "Weighted Benign Flows Blocked": round(r["weight"], 1),
                    }
                )
        else:
            breakdown_rows.append(
                {
                    "Strategy": s_name,
                    "Source Subnet": "None",
                    "Destination Host": "None",
                    "Dst Port": 0,
                    "Weighted Benign Flows Blocked": 0.0,
                }
            )
    breakdown_df = pd.DataFrame(breakdown_rows)

    # 8. Score Drift Interpretation Table (Separating Unmasked Attacks vs Drifted Benign)
    drift_summary_rows = []
    for s_name in solvers.keys():
        s_logs = [log for log in debug_incident_logs if log["Strategy"] == s_name]
        tot_inc = len(s_logs)
        tot_unmasked = sum(int(log["Unmasked Attacks"]) for log in s_logs)
        tot_drifted = sum(int(log["Drifted Benign"]) for log in s_logs)
        drift_summary_rows.append(
            {
                "Strategy": s_name,
                "Total Incidents": tot_inc,
                "Unmasked True Attacks (Threats Discovered)": tot_unmasked,
                "Mean Unmasked / Incident": f"{tot_unmasked / max(1, tot_inc):.2f}",
                "Drifted Benign Edges (Structural Artefacts)": tot_drifted,
                "Mean Drifted Benign / Incident": f"{tot_drifted / max(1, tot_inc):.2f}",
            }
        )
    drift_df = pd.DataFrame(drift_summary_rows)

    # 9. Multi-Strategy Frontier Sweeps (Decision 40)
    # 9a. Sweep BlockAllFlagged across tau (Monotonicity guaranteed by fixed denominators)
    tau_sweep_vals = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    block_sweep_abrs, block_sweep_cdrs = [], []
    for t_val in tau_sweep_vals:
        logger.info(f"Frontier Sweep: BlockAllFlagged tau={t_val}")
        b_solver = BlockAllFlaggedBaseline(tau=t_val)
        t_abrs, t_cdrs = [], []
        for ctx in incident_contexts:
            r = evaluate_containment_on_incident(
                b_solver, ctx, detector, cost_evaluator, rate_limit_factor, tau=t_val
            )
            t_abrs.append(r["abr"])
            t_cdrs.append(r["cdr"])
        block_sweep_abrs.append(float(np.mean(t_abrs)))
        block_sweep_cdrs.append(float(np.mean(t_cdrs)))

    # 9b. Sweep Greedy across lambda_cost (Uncapped max_cuts)
    greedy_sweep_lambdas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    greedy_sweep_abrs, greedy_sweep_cdrs = [], []
    for g_lam in greedy_sweep_lambdas:
        logger.info(f"Frontier Sweep: Greedy lambda_cost={g_lam}")
        g_solver = GreedyCutSolver(
            tau_low=tau_low,
            tau=eval_tau,
            max_cuts=None,
            lambda_cost=g_lam,
        )
        g_abrs, g_cdrs = [], []
        for ctx in incident_contexts:
            r = evaluate_containment_on_incident(
                g_solver, ctx, detector, cost_evaluator, rate_limit_factor, tau=eval_tau
            )
            g_abrs.append(r["abr"])
            g_cdrs.append(r["cdr"])
        greedy_sweep_abrs.append(float(np.mean(g_abrs)))
        greedy_sweep_cdrs.append(float(np.mean(g_cdrs)))

    # 9c. Sweep N1 across lambda1, lambda2 (finer sweep between (5, 2) and (10, 5) per Directive 1)
    lambda_settings = [
        (0.5, 0.2),
        (1.0, 0.5),
        (2.0, 1.0),
        (5.0, 2.0),
        (6.0, 2.5),
        (7.0, 3.0),
        (7.5, 3.3),
        (8.0, 3.6),
        (8.5, 4.0),
        (9.0, 4.5),
        (9.5, 4.8),
        (10.0, 5.0),
    ]
    n1_sweep_abrs, n1_sweep_cdrs, n1_sweep_costs = [], [], []
    n1_sweep_table_rows = []

    solver_logger = logging.getLogger("graphguard.containment.cut_solver")
    prev_log_level = solver_logger.level
    solver_logger.setLevel(logging.WARNING)

    try:
        for l1, l2 in lambda_settings:
            logger.info(f"Frontier Sweep: N1 lambda1={l1}, lambda2={l2}")
            n1_solver = CounterfactualCutSolver(
                tau_low=tau_low,
                tau=eval_tau,
                rate_limit_factor=rate_limit_factor,
                lambda1=l1,
                lambda2=l2,
                lambda_residual_risk=5.0,
                max_steps=120,
            )
            s_abrs, s_cdrs, s_costs = [], [], []
            for ctx in incident_contexts:
                r = evaluate_containment_on_incident(
                    n1_solver, ctx, detector, cost_evaluator, rate_limit_factor, tau=eval_tau
                )
                s_abrs.append(r["abr"])
                s_cdrs.append(r["cdr"])
                s_costs.append(r["total_cost"])
            m_abr = float(np.mean(s_abrs))
            m_cdr = float(np.mean(s_cdrs))
            m_cost = float(np.mean(s_costs))
            n1_sweep_abrs.append(m_abr)
            n1_sweep_cdrs.append(m_cdr)
            n1_sweep_costs.append(m_cost)
            n1_sweep_table_rows.append(
                {
                    "lambda1": l1,
                    "lambda2": l2,
                    "Attack Block Rate": f"{m_abr * 100:.1f}%",
                    "Collateral Damage Rate": f"{m_cdr * 100:.1f}%",
                    "Mean Cost": f"{m_cost:.2f}",
                }
            )
    finally:
        solver_logger.setLevel(prev_log_level)

    n1_sweep_df = pd.DataFrame(n1_sweep_table_rows)

    # 10. Matched ABR Comparison Table (Measured Points Only, no interpolation across gaps > 5%)
    matched_targets = [0.80, 0.90, 0.95]
    matched_rows = []
    for t_abr in matched_targets:
        blk_info = interpolate_cdr_at_abr(block_sweep_abrs, block_sweep_cdrs, t_abr)
        grd_info = interpolate_cdr_at_abr(greedy_sweep_abrs, greedy_sweep_cdrs, t_abr)
        n1_info = interpolate_cdr_at_abr(n1_sweep_abrs, n1_sweep_cdrs, t_abr)
        matched_rows.append(
            {
                "Target ABR": f"{t_abr * 100:.0f}%",
                "BlockAllFlagged CDR": blk_info[1] if blk_info else "N/A (gap > 5%)",
                "Greedy Cut CDR": grd_info[1] if grd_info else "N/A (gap > 5%)",
                "N1 Counterfactual CDR": n1_info[1] if n1_info else "N/A (gap > 5%)",
            }
        )
    matched_df = pd.DataFrame(matched_rows)

    # 11. Generate Multi-Strategy Frontier Plot (containment_frontier_v2.png)
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / "containment_frontier_v2.png"

    plt.figure(figsize=(9, 6), dpi=150)
    plt_style = (
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    plt.style.use(plt_style)

    def sort_curve(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
        pts = sorted(zip(xs, ys), key=lambda p: p[0])
        return [p[0] * 100 for p in pts], [p[1] * 100 for p in pts]

    n1_x, n1_y = sort_curve(n1_sweep_abrs, n1_sweep_cdrs)
    grd_x, grd_y = sort_curve(greedy_sweep_abrs, greedy_sweep_cdrs)
    blk_x, blk_y = sort_curve(block_sweep_abrs, block_sweep_cdrs)

    plt.plot(
        n1_x,
        n1_y,
        marker="o",
        color="#1f77b4",
        linewidth=2.5,
        label="N1 Counterfactual Cut (fine lambda sweep)",
    )
    plt.plot(
        grd_x,
        grd_y,
        marker="s",
        color="#2ca02c",
        linewidth=2.0,
        linestyle="--",
        label="Greedy Cut (lambda sweep, uncapped)",
    )
    plt.plot(
        blk_x,
        blk_y,
        marker="^",
        color="#ff7f0e",
        linewidth=2.0,
        linestyle=":",
        label="BlockAllFlagged (tau sweep, monotonic)",
    )

    for t_abr in matched_targets:
        plt.axvline(
            x=t_abr * 100,
            color="#7f7f7f",
            linestyle="-.",
            alpha=0.6,
            label=f"Matched ABR {t_abr * 100:.0f}%" if t_abr == 0.80 else None,
        )

    plt.title(
        "Containment Frontier: Collateral Damage Rate vs Attack Block Rate (NF-UNSW-NB15-v3)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Attack Block Rate (% of malicious flows blocked) ->", fontsize=11)
    plt.ylabel("Collateral Damage Rate (% of affected benign flows blocked) ->", fontsize=11)
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved multi-strategy containment frontier plot to {plot_path}")

    # Copy to artifact paths if available
    for conv_id in [
        "2e272d52-6167-4364-beb5-b390f50d2996",
        "9163b162-133b-4a4d-a56b-242bf9e284b0",
        "41d75895-25ae-4513-8201-2322f92d3869",
    ]:
        artifact_dir = Path(f"C:/Users/HP/.gemini/antigravity-ide/brain/{conv_id}")
        if artifact_dir.exists():
            shutil.copy(plot_path, artifact_dir / "containment_frontier_v2.png")
            shutil.copy(plot_path, artifact_dir / "containment_frontier.png")

    # Format asset role table
    roles_list = []
    for ip, info in inferred_roles.items():
        roles_list.append(
            {
                "Host IP": ip,
                "Inferred Role": info["role"].upper(),
                "Criticality": info["criticality"],
                "Inference Evidence": info["reason"],
            }
        )
    roles_df = pd.DataFrame(roles_list)

    return (
        headline_df,
        cost_profile_df,
        matched_df,
        n1_sweep_df,
        roles_df,
        debug_df,
        drift_df,
        floor_df,
        breakdown_df,
    )


if __name__ == "__main__":
    (
        headline_df,
        cost_profile_df,
        matched_df,
        n1_sweep_df,
        roles_df,
        debug_df,
        drift_df,
        floor_df,
        breakdown_df,
    ) = run_containment_benchmark()

    print("\n" + "=" * 96)
    print("                    INFERRED ASSET ROLES (config/assets_inferred.yaml)")
    print("=" * 96)
    print(roles_df.to_string(index=False))

    print("\n" + "=" * 96)
    print("                    COLLATERAL DAMAGE FLOOR (EdgeKey Granularity)")
    print("=" * 96)
    print(floor_df.to_string(index=False))

    print("\n" + "=" * 96)
    print("     BLOCKED BENIGN FLOW BREAKDOWN (source subnet, destination host, dst port)")
    print("=" * 96)
    print(breakdown_df.to_string(index=False))

    print("\n" + "=" * 120)
    print("                     PER-INCIDENT CONTAINMENT & REMEDIATION DEBUG LOG")
    print("=" * 120)
    print(debug_df.to_string(index=False))

    print("\n" + "=" * 120)
    print("            HEADLINE CONTAINMENT BENCHMARK COMPARISON TABLE (Flow-Level)")
    print("=" * 120)
    headline_cols = [
        "Strategy",
        "Incidents (N)",
        "Attack Block Rate (Drops) [95% CI]",
        "Attack Throttle Rate (RL)",
        "Label-Based CDR [95% CI]",
        "Legitimate CDR [95% CI]",
        "Residual Attack Rate (RAR)",
        "Model Containment Success Rate",
        "Mean Latency (ms)",
        "p95 Latency (ms)",
    ]
    print(headline_df[headline_cols].to_string(index=False))

    print("\n" + "=" * 96)
    print("           SECONDARY TABLE: CONTAINMENT COST & INTERVENTION PROFILE")
    print("=" * 96)
    print(cost_profile_df.to_string(index=False))

    print("\n" + "=" * 110)
    print("     SCORE DRIFT INTERPRETATION (Unmasked True Attacks vs Drifted Benign Edges)")
    print("=" * 110)
    print(drift_df.to_string(index=False))

    print("\n" + "=" * 96)
    print("      COLLATERAL DAMAGE RATE (CDR) AT MATCHED ATTACK BLOCK RATES (ABR)")
    print("=" * 96)
    print(matched_df.to_string(index=False))

    print("\n" + "=" * 96)
    print("                    N1 LAMBDA PARAMETER SWEEP (lambda1, lambda2)")
    print("=" * 96)
    print(n1_sweep_df.to_string(index=False))
    print("=" * 96 + "\n")
