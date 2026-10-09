"""End-to-end training, calibration, baseline benchmarking, and evaluation pipeline for Phase P1."""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam

from graphguard.detect.egraphsage import EGraphSAGEDetector, EGraphSAGENet
from graphguard.detect.open_set import (
    OpenSetEvaluator,
    calibrate_energy_threshold,
    calibrate_tau,
    compute_energy_scores,
)
from graphguard.detect.registry import ModelRegistry
from graphguard.eval.baselines import TabularRandomForestBaseline
from graphguard.eval.metrics import (
    compute_binary_metrics,
    compute_classification_metrics,
    evaluate_hub_dependence,
)
from graphguard.graph.features import NetFlowFeatureExtractor
from graphguard.graph.window_builder import inspect_timestamp_column

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "config/detection.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_dataset_file(cfg: Dict[str, Any]) -> Optional[Path]:
    raw_dir = Path(cfg.get("data", {}).get("raw_dir", "data/raw"))
    target_path = Path(cfg.get("data", {}).get("dataset_path", "data/raw/NF-UNSW-NB15-v2.csv"))

    if target_path.exists():
        return target_path

    if raw_dir.exists():
        csvs = list(raw_dir.glob("*.csv"))
        if csvs:
            return csvs[0]

    sample_path = Path(cfg.get("data", {}).get("sample_path", "data/sample_nf_v2.csv"))
    if sample_path.exists():
        return sample_path

    return None


def split_dataframe_time_ordered(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform strictly time-ordered split preserving row order."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def dataframe_to_pyg_data(
    df: pd.DataFrame,
    features_matrix: np.ndarray,
    global_node_map: Optional[Dict[str, int]] = None,
) -> Tuple[Any, List[str], Dict[str, int]]:
    """Build PyG Data object directly from DataFrame and features without memory bloat."""
    from torch_geometric.data import Data

    lookup = {c.lower().strip(): c for c in df.columns}
    src_col = lookup.get("ipv4_src_addr") or lookup.get("src_ip")
    dst_col = lookup.get("ipv4_dst_addr") or lookup.get("dst_ip")

    src_ips = df[src_col].astype(str).str.strip().to_numpy()
    dst_ips = df[dst_col].astype(str).str.strip().to_numpy()

    if global_node_map is None:
        node_ids = sorted(list(np.unique(np.concatenate([src_ips, dst_ips]))))
        ip_to_idx = {ip: i for i, ip in enumerate(node_ids)}
    else:
        ip_to_idx = global_node_map
        node_ids = [ip for ip, _ in sorted(ip_to_idx.items(), key=lambda x: x[1])]

    u = np.array([ip_to_idx[ip] for ip in src_ips], dtype=np.int64)
    v = np.array([ip_to_idx[ip] for ip in dst_ips], dtype=np.int64)

    edge_index = torch.tensor(np.vstack([u, v]), dtype=torch.long)
    edge_attr = torch.from_numpy(features_matrix.astype(np.float32))
    x = torch.ones((len(node_ids), 1), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, node_ids, ip_to_idx


def extract_class_mapping(df: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
    """Derive multi-class labels: Benign is class 0, attack types are 1..C."""
    attack_col = None
    for c in df.columns:
        if c.lower().strip() == "attack":
            attack_col = c
            break

    if not attack_col:
        for c in df.columns:
            if c.lower().strip() == "label":
                attack_col = c
                break

    if attack_col:
        unique_vals = [str(v).strip() for v in df[attack_col].unique()]
        # Put benign / 0 first
        benign_matches = [v for v in unique_vals if v.lower() in ("benign", "0")]
        non_benign = sorted([v for v in unique_vals if v.lower() not in ("benign", "0")])

        if benign_matches:
            classes = ["Benign"] + non_benign
        else:
            classes = ["Benign"] + unique_vals
    else:
        classes = ["Benign", "Attack"]

    class_to_idx = {name: i for i, name in enumerate(classes)}
    return class_to_idx, classes


def map_labels_to_ids(
    df: pd.DataFrame,
    class_to_idx: Dict[str, int],
) -> np.ndarray:
    attack_col = None
    for c in df.columns:
        if c.lower().strip() == "attack":
            attack_col = c
            break
    if not attack_col:
        for c in df.columns:
            if c.lower().strip() == "label":
                attack_col = c
                break

    if not attack_col:
        return np.zeros(len(df), dtype=int)

    mapped = []
    for val in df[attack_col]:
        v_str = str(val).strip()
        if v_str.lower() in ("benign", "0"):
            mapped.append(0)
        elif v_str in class_to_idx:
            mapped.append(class_to_idx[v_str])
        else:
            mapped.append(1 if len(class_to_idx) > 1 else 0)

    return np.array(mapped, dtype=int)


def train_egraphsage_model(
    train_win_data: Any,
    y_train: np.ndarray,
    val_win_data: Any,
    y_val: np.ndarray,
    edge_dim: int,
    num_classes: int,
    epochs: int = 8,
    lr: float = 0.005,
    hidden_dim: int = 64,
    dropout: float = 0.1,
    weight_decay: float = 0.0001,
    device: str = "cpu",
) -> Tuple[EGraphSAGENet, Dict[str, List[float]]]:
    """Train EGraphSAGENet with class-imbalance weighting and track loss curves."""
    net = EGraphSAGENet(
        edge_dim=edge_dim,
        num_classes=num_classes,
        node_dim=1,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    # Class weighting
    class_counts = np.bincount(y_train, minlength=num_classes)
    total_samples = len(y_train)
    weights = []
    for c in class_counts:
        weights.append(total_samples / (num_classes * max(c, 1)))
    weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    x_tr = train_win_data.x.to(device)
    edge_idx_tr = train_win_data.edge_index.to(device)
    edge_attr_tr = train_win_data.edge_attr.to(device)
    y_tr_tensor = torch.tensor(y_train, dtype=torch.long, device=device)

    x_val = val_win_data.x.to(device)
    edge_idx_val = val_win_data.edge_index.to(device)
    edge_attr_val = val_win_data.edge_attr.to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=device)

    curves: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_f1": []}

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad()
        out_tr = net(x_tr, edge_idx_tr, edge_attr_tr)
        loss_tr = criterion(out_tr, y_tr_tensor)
        loss_tr.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            out_val = net(x_val, edge_idx_val, edge_attr_val)
            loss_val = criterion(out_val, y_val_tensor)
            val_preds = out_val.argmax(dim=-1).cpu().numpy()
            y_val_bin = (y_val > 0).astype(int)
            val_preds_bin = (val_preds > 0).astype(int)
            val_f1 = float(np.mean(val_preds_bin == y_val_bin))

        curves["train_loss"].append(float(loss_tr.item()))
        curves["val_loss"].append(float(loss_val.item()))
        curves["val_f1"].append(val_f1)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {loss_tr.item():.4f} | "
            f"Val Loss: {loss_val.item():.4f} | "
            f"Val Acc: {val_f1:.4f}"
        )

        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            best_weights = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    if best_weights:
        net.load_state_dict(best_weights)

    return net, curves


def run_pipeline() -> None:
    """Execute complete Phase P1 pipeline."""
    cfg = load_config("config/detection.yaml")
    set_seed(cfg.get("data", {}).get("seed", 42))

    dataset_file = find_dataset_file(cfg)
    if not dataset_file:
        logger.error(
            "No NetFlow dataset found in data/raw/ or data/sample_nf_v2.csv. "
            "Please download/place the CSV dataset into data/raw/."
        )
        sys.exit(1)

    logger.info(f"Loading dataset from: {dataset_file}")
    df = pd.read_csv(dataset_file)
    logger.info(f"Loaded {len(df):,} flows with {len(df.columns)} columns.")

    # 1. Timestamp inspection (Decision 15, Decision 28)
    has_time, time_col = inspect_timestamp_column(df)
    if has_time and time_col:
        logger.info(
            f"[Timestamp Check] Found timestamp column '{time_col}'. "
            f"Sorting dataset chronologically for time-ordered split."
        )
        df = df.sort_values(by=time_col).reset_index(drop=True)
    else:
        logger.info(
            "[Timestamp Check] No timestamp column found. Time-ordered split assumes "
            "CSV row order is chronological, which cannot be verified."
        )

    # Dataset Accounting & Provenance (Requirement 2)
    lookup_cols = {c.lower().strip(): c for c in df.columns}
    src_c = lookup_cols.get("ipv4_src_addr") or lookup_cols.get("src_ip")
    dst_c = lookup_cols.get("ipv4_dst_addr") or lookup_cols.get("dst_ip")
    unique_src_ips = sorted(list(df[src_c].astype(str).unique())) if src_c else []
    unique_dst_ips = sorted(list(df[dst_c].astype(str).unique())) if dst_c else []
    all_ips = sorted(list(set(unique_src_ips) | set(unique_dst_ips)))

    print("\n" + "=" * 60)
    print("DATASET PROVENANCE AND STATISTICAL ACCOUNTING:")
    print(f"  Source File:           {dataset_file}")
    print(f"  Total Flow Count:      {len(df):,}")
    print(f"  Total Columns:         {len(df.columns)}")
    src_sample = f"{unique_src_ips[:3]} ... {unique_src_ips[-3:]}"
    dst_sample = f"{unique_dst_ips[:3]} ... {unique_dst_ips[-3:]}"
    print(f"  Unique Source IPs:     {len(unique_src_ips)} ({src_sample})")
    print(f"  Unique Dest IPs:       {len(unique_dst_ips)} ({dst_sample})")
    print(f"  Total Network Nodes:   {len(all_ips)}")
    if has_time and time_col:
        min_ts = df[time_col].min()
        max_ts = df[time_col].max()
        divisor = 1000.0 if max_ts > 1e11 else 1.0
        t0_dt = datetime.fromtimestamp(min_ts / divisor, tz=timezone.utc)
        t1_dt = datetime.fromtimestamp(max_ts / divisor, tz=timezone.utc)
        span_h = (max_ts - min_ts) / (divisor * 3600.0)
        print(f"  Timestamp Column:      {time_col}")
        print(f"  Time Span:             {t0_dt} to {t1_dt} ({span_h:.2f} hours)")
    print("=" * 60 + "\n")

    # 2. Extract stratified sample for unit tests if not present
    sample_path = Path(cfg.get("data", {}).get("sample_path", "data/sample_nf_v2.csv"))
    if not sample_path.exists() and len(df) > 2000:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        # Stratified slice of 5,000 flows
        label_col = (
            "Label" if "Label" in df.columns else ("Attack" if "Attack" in df.columns else None)
        )
        if label_col:
            sample_df = (
                df.groupby(label_col, group_keys=False)
                .apply(lambda x: x.sample(min(len(x), 500), random_state=42))
                .head(5000)
            )
        else:
            sample_df = df.head(5000)
        sample_df.to_csv(sample_path, index=False)
        logger.info(f"Saved stratified unit-test slice of {len(sample_df)} rows to {sample_path}")

    # 3. Time-ordered split (Decision 15)
    train_ratio = cfg.get("data", {}).get("train_ratio", 0.70)
    val_ratio = cfg.get("data", {}).get("val_ratio", 0.15)
    train_df, val_df, test_df = split_dataframe_time_ordered(df, train_ratio, val_ratio)
    logger.info(
        f"Time-ordered Split: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}"
    )

    # 4. Feature extraction (Decision 16)
    feature_extractor = NetFlowFeatureExtractor()
    feature_extractor.fit(train_df)
    feature_list = feature_extractor.feature_names
    print("\n" + "=" * 60)
    print(f"FINAL NETFLOW NUMERIC FEATURE LIST ({len(feature_list)} features):")
    for f_idx, feat in enumerate(feature_list, 1):
        print(f"  {f_idx:02d}. {feat}")
    print("=" * 60 + "\n")

    X_train = feature_extractor.transform(train_df)
    X_val = feature_extractor.transform(val_df)
    X_test = feature_extractor.transform(test_df)

    # 5. Label mapping & class counts (Decision 21, Decision 25)
    class_to_idx, class_names = extract_class_mapping(df)
    y_train = map_labels_to_ids(train_df, class_to_idx)
    y_val = map_labels_to_ids(val_df, class_to_idx)
    y_test = map_labels_to_ids(test_df, class_to_idx)

    y_train_bin = (y_train > 0).astype(int)
    y_val_bin = (y_val > 0).astype(int)
    y_test_bin = (y_test > 0).astype(int)

    print("\n" + "=" * 60)
    print("CLASS COUNTS PER SPLIT:")
    for c_idx, c_name in enumerate(class_names):
        n_tr = int(np.sum(y_train == c_idx))
        n_va = int(np.sum(y_val == c_idx))
        n_te = int(np.sum(y_test == c_idx))
        print(f"  Class '{c_name}': Train={n_tr:,} | Val={n_va:,} | Test={n_te:,}")
    print("=" * 60 + "\n")

    # 6. Build PyG Data objects for GNN
    global_node_map = {ip: i for i, ip in enumerate(all_ips)}
    pyg_tr, _, _ = dataframe_to_pyg_data(train_df, X_train, global_node_map)
    pyg_val, _, _ = dataframe_to_pyg_data(val_df, X_val, global_node_map)
    pyg_test, _, _ = dataframe_to_pyg_data(test_df, X_test, global_node_map)

    # 7. Train E-GraphSAGE
    model_hp = cfg.get("model", {})
    epochs = model_hp.get("epochs", 8)
    hidden_dim = model_hp.get("hidden_dim", 64)
    dropout = model_hp.get("dropout", 0.1)
    lr = model_hp.get("lr", 0.005)

    logger.info("Training E-GraphSAGE model...")
    net, training_curves = train_egraphsage_model(
        train_win_data=pyg_tr,
        y_train=y_train,
        val_win_data=pyg_val,
        y_val=y_val,
        edge_dim=len(feature_list),
        num_classes=len(class_names),
        epochs=epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    # 8. Calibration on Validation Split (Decision 19, Decision 26)
    net.eval()
    with torch.no_grad():
        val_logits = net(pyg_val.x, pyg_val.edge_index, pyg_val.edge_attr).cpu().numpy()
        val_probs = torch.softmax(torch.from_numpy(val_logits), dim=-1).numpy()
        val_attack_scores = np.clip(1.0 - val_probs[:, 0], 0.0, 1.0)

    best_tau, val_f1_tau = calibrate_tau(y_val_bin, val_attack_scores)

    known_val_attack_energies = compute_energy_scores(val_logits[y_val_bin == 1])
    energy_threshold = calibrate_energy_threshold(known_val_attack_energies)

    detector = EGraphSAGEDetector(
        model=net,
        classes=class_names,
        tau=best_tau,
        energy_threshold=energy_threshold,
    )

    # 9. Evaluate E-GraphSAGE on Test Split
    with torch.no_grad():
        test_logits = net(pyg_test.x, pyg_test.edge_index, pyg_test.edge_attr).cpu().numpy()
        test_probs = torch.softmax(torch.from_numpy(test_logits), dim=-1).numpy()
        test_attack_scores = np.clip(1.0 - test_probs[:, 0], 0.0, 1.0)
        test_preds_multi = test_logits.argmax(axis=-1)

    gnn_bin_metrics = compute_binary_metrics(y_test_bin, test_attack_scores, tau=best_tau)
    gnn_multi_metrics = compute_classification_metrics(y_test, test_preds_multi, class_names)

    # 10. Train Tabular Baseline (RandomForest) (Decision 18, Decision 27)
    logger.info("Training Tabular RandomForest baseline on identical edge features...")
    rf_baseline = TabularRandomForestBaseline(random_state=42)
    max_rf_samples = cfg.get("baseline", {}).get("max_train_samples", 50000)
    rf_baseline.fit(X_train, y_train_bin, max_samples=max_rf_samples)
    rf_metrics = rf_baseline.evaluate(X_test, y_test_bin)

    print("\n" + "=" * 60)
    print("DETECTOR PERFORMANCE COMPARISON (Test Split):")
    print(f"{'Metric':<20} | {'E-GraphSAGE (GNN)':<18} | {'RandomForest (Baseline)':<22}")
    p_gnn, p_rf = gnn_bin_metrics["binary_precision"], rf_metrics["precision"]
    r_gnn, r_rf = gnn_bin_metrics["binary_recall"], rf_metrics["recall"]
    f1_gnn, f1_rf = gnn_bin_metrics["binary_f1"], rf_metrics["f1"]
    m_f1 = gnn_multi_metrics["macro_f1"]
    print(f"{'Binary Precision':<20} | {p_gnn:<18.4f} | {p_rf:<22.4f}")
    print(f"{'Binary Recall':<20} | {r_gnn:<18.4f} | {r_rf:<22.4f}")
    print(f"{'Binary F1':<20} | {f1_gnn:<18.4f} | {f1_rf:<22.4f}")
    print(f"{'Macro-F1 (multi)':<20} | {m_f1:<18.4f} | {'N/A (binary)':<22}")
    print(f"{'Calibrated tau':<20} | {best_tau:<18.3f} | {'0.500 (default)':<22}")
    print("=" * 60 + "\n")

    # Print multi-class breakdown
    print("\n" + "=" * 60)
    print("E-GRAPHSAGE PER-CLASS F1 BREAKDOWN:")
    for c_name, sc in gnn_multi_metrics["per_class"].items():
        print(
            f"  {c_name:<16}: Prec={sc['precision']:.4f}, Rec={sc['recall']:.4f}, "
            f"F1={sc['f1']:.4f}, Supp={sc['support']}"
        )
    print("=" * 60 + "\n")

    # 11. Per-Flow Source Randomization Hub-Dependence Check (Decision 24)
    logger.info("Running per-flow source randomization hub-dependence check...")
    hub_dep = evaluate_hub_dependence(
        test_df=test_df,
        feature_extractor=feature_extractor,
        detector=detector,
        original_f1=gnn_bin_metrics["binary_f1"],
    )
    print("\n" + "=" * 60)
    print("HUB-DEPENDENCE CHECK (Per-Flow Source Randomization):")
    print(f"  Original F1:            {hub_dep['original_f1']:.4f}")
    print(f"  Randomized Source F1:   {hub_dep['randomized_f1']:.4f}")
    print(f"  Delta:                  {hub_dep['f1_delta']:+.4f}")
    print("=" * 60 + "\n")

    # 12. Leave-One-Attack-Out Open-Set Evaluation (Decision 20, Decision 25)
    logger.info("Running leave-one-attack-out open-set evaluation...")
    evaluator = OpenSetEvaluator(target_holdout="Reconnaissance")
    holdout_class = evaluator.select_holdout_class(class_names)

    if holdout_class and holdout_class in class_to_idx:
        holdout_idx = class_to_idx[holdout_class]
        logger.info(
            f"Held out attack category for open-set AUROC: '{holdout_class}' (idx={holdout_idx})"
        )

        in_dist_mask = y_test != holdout_idx
        ood_mask = y_test == holdout_idx

        if np.sum(ood_mask) > 0 and np.sum(in_dist_mask) > 0:
            open_set_res = evaluator.evaluate_open_set_auroc(
                in_dist_logits=test_logits[in_dist_mask],
                ood_logits=test_logits[ood_mask],
            )
            print("\n" + "=" * 60)
            print(f"LEAVE-ONE-ATTACK-OUT OPEN-SET EVALUATION (Held-out: '{holdout_class}'):")
            print(f"  UNKNOWN Detection AUROC:     {open_set_res['open_set_auroc']:.4f}")
            print(
                f"  Mean In-Distribution Energy:  {open_set_res['mean_in_distribution_energy']:.4f}"
            )
            print(f"  Mean UNKNOWN Attack Energy:   {open_set_res['mean_ood_energy']:.4f}")
            print("=" * 60 + "\n")
        else:
            open_set_res = {
                "open_set_auroc": 0.50,
                "note": "Insufficient samples for heldout class",
            }
    else:
        open_set_res = {
            "open_set_auroc": 0.50,
            "note": "No distinct attack category found to hold out",
        }

    # 13. Persist to Model Registry (Decision 23)
    registry = ModelRegistry("models")
    metrics_bundle = {
        "binary_gnn": gnn_bin_metrics,
        "multi_class_gnn": gnn_multi_metrics,
        "baseline_rf": rf_metrics,
        "hub_dependence": hub_dep,
        "open_set": open_set_res,
    }
    hp_bundle = {
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "lr": lr,
        "mc_dropout": False,
    }
    version = registry.save_model(
        model_name="egraphsage",
        model=net,
        feature_extractor=feature_extractor,
        classes=class_names,
        calibrated_tau=best_tau,
        energy_threshold=energy_threshold,
        metrics=metrics_bundle,
        training_curves=training_curves,
        hyperparameters=hp_bundle,
    )
    logger.info(f"Phase P1 Complete! Model registered as 'egraphsage' version '{version}'.")


if __name__ == "__main__":
    run_pipeline()
