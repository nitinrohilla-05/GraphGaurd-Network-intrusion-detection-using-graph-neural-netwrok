"""Free energy score, threshold calibration, and leave-one-attack-out open-set evaluation."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

logger = logging.getLogger(__name__)


def compute_energy_scores(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute negative free energy scores: E(x; T) = -T * logsumexp(logits / T).

    In-distribution samples produce lower energy; out-of-distribution (UNKNOWN)
    novel attacks produce higher energy.
    """
    scaled_logits = logits / max(temperature, 1e-6)
    # LogSumExp trick for numerical stability
    max_l = np.max(scaled_logits, axis=-1, keepdims=True)
    lse = max_l.squeeze(-1) + np.log(np.sum(np.exp(scaled_logits - max_l), axis=-1))
    energy = -temperature * lse
    return energy


def calibrate_tau(
    y_true_binary: np.ndarray,
    attack_scores: np.ndarray,
    search_steps: int = 100,
) -> Tuple[float, float]:
    """Calibrate decision threshold tau on validation set by maximizing binary F1 score.

    Returns: (best_tau, best_f1)
    """
    best_tau = 0.50
    best_f1 = 0.0

    thresholds = np.linspace(0.05, 0.95, search_steps)
    for tau in thresholds:
        preds = (attack_scores >= tau).astype(int)
        score = float(f1_score(y_true_binary, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_tau = float(tau)

    logger.info(f"Calibrated threshold tau={best_tau:.3f} (Validation Binary F1={best_f1:.4f})")
    return best_tau, best_f1


def calibrate_energy_threshold(
    known_attack_energies: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """Calibrate energy threshold at target True Positive Rate on known validation attacks."""
    if len(known_attack_energies) == 0:
        return -5.0
    # Higher energy is considered OOD/UNKNOWN.
    # Set threshold at the (target_tpr * 100)-th percentile
    threshold = float(np.percentile(known_attack_energies, target_tpr * 100))
    logger.info(f"Calibrated open-set energy threshold={threshold:.3f} (TPR={target_tpr:.2f})")
    return threshold


class OpenSetEvaluator:
    """Evaluates open-set detection using leave-one-attack-out experiments."""

    def __init__(self, target_holdout: str = "Reconnaissance") -> None:
        self.target_holdout = target_holdout

    def select_holdout_class(self, attack_categories: List[str]) -> Optional[str]:
        """Find target holdout class or closest matching category from real dataset categories."""
        # Exact case-insensitive match
        for cat in attack_categories:
            if cat.strip().lower() == self.target_holdout.lower():
                return cat

        # Substring match (e.g. "Recon" in "Reconnaissance")
        for cat in attack_categories:
            if "recon" in cat.strip().lower():
                return cat

        # Non-benign candidate fallback
        non_benign = [c for c in attack_categories if c.strip().lower() not in ("benign", "0")]
        return non_benign[0] if non_benign else None

    def evaluate_open_set_auroc(
        self,
        in_dist_logits: np.ndarray,
        ood_logits: np.ndarray,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Compute AUROC for distinguishing held-out attack (OOD) via energy scores."""
        in_energies = compute_energy_scores(in_dist_logits, temperature)
        ood_energies = compute_energy_scores(ood_logits, temperature)

        y_true = np.concatenate([np.zeros(len(in_energies)), np.ones(len(ood_energies))])
        y_scores = np.concatenate([in_energies, ood_energies])

        try:
            auroc = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            auroc = 0.50

        return {
            "open_set_auroc": auroc,
            "mean_in_distribution_energy": float(np.mean(in_energies)),
            "mean_ood_energy": float(np.mean(ood_energies)),
        }
