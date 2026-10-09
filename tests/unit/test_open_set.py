"""Unit tests for free energy score, calibration, and open-set evaluator."""

import numpy as np

from graphguard.detect.open_set import (
    OpenSetEvaluator,
    calibrate_energy_threshold,
    calibrate_tau,
    compute_energy_scores,
)


def test_compute_energy_scores() -> None:
    """Larger logits yield lower energy; diffuse/small logits yield higher energy."""
    # Strong in-distribution prediction
    logits_in = np.array([[10.0, 1.0], [8.0, 0.5]])
    # Ambiguous / zero-like prediction
    logits_out = np.array([[0.1, 0.1], [-1.0, -1.0]])

    e_in = compute_energy_scores(logits_in, temperature=1.0)
    e_out = compute_energy_scores(logits_out, temperature=1.0)

    # In-distribution energies should be significantly lower (more negative)
    assert np.mean(e_in) < np.mean(e_out)


def test_calibrate_tau() -> None:
    """Calibrate tau selects threshold maximizing binary F1 score."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    attack_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.75, 0.3, 0.85, 0.15])

    best_tau, best_f1 = calibrate_tau(y_true, attack_scores)
    assert 0.3 < best_tau < 0.75
    assert best_f1 == 1.0


def test_calibrate_energy_threshold() -> None:
    """Calibrate energy threshold sets percentile cutoff."""
    energies = np.array([-10.0, -8.0, -6.0, -4.0, -2.0])
    thresh = calibrate_energy_threshold(energies, target_tpr=0.80)
    assert -5.0 <= thresh <= -2.0


def test_open_set_evaluator_holdout_selection() -> None:
    """Selects Reconnaissance or closest available category."""
    evaluator = OpenSetEvaluator(target_holdout="Reconnaissance")
    cats = ["Benign", "DoS", "Reconnaissance", "Exploits"]
    chosen = evaluator.select_holdout_class(cats)
    assert chosen == "Reconnaissance"

    cats_substring = ["Benign", "DoS", "Recon", "Worms"]
    chosen_sub = evaluator.select_holdout_class(cats_substring)
    assert chosen_sub == "Recon"
