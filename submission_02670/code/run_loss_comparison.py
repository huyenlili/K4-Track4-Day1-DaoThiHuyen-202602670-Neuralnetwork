"""Run the MSE baseline and save validation results and comparison plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import prepare_data
from plots import plot_compare, plot_run
from train import run_experiment
from results_table import save_result


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parents[1]
OUTPUT_DIR = REPO_ROOT / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=OUTPUT_DIR / "data" / "processed",
        help="Directory containing train.npz and eval.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory in which results and figures are saved.",
    )
    args = parser.parse_args()

    for filename in ("train.npz", "eval.npz"):
        if not (args.processed_dir / filename).is_file():
            raise FileNotFoundError(
                f"Missing {args.processed_dir / filename}; run scripts/split_data.py first."
            )

    results_dir = args.output_dir / "results"
    ce_path = results_dir / "loss-ce-s0.json"
    if not ce_path.is_file():
        ce_path = results_dir / "base-s0.json"
    if not ce_path.is_file():
        raise FileNotFoundError(
            f"Missing CE result in {results_dir}; run the baseline experiment first."
        )

    with ce_path.open("r", encoding="utf-8") as file:
        ce_result = json.load(file)

    cfg = ce_result["cfg"]
    if cfg.get("loss") != "ce" or cfg.get("seed") != 0:
        raise ValueError("The comparison result must be the CE baseline with seed 0.")
    expected = {
        "optimizer": "sgd_momentum",
        "lr": 0.1,
        "weight_decay": 0.0,
        "momentum": 0.9,
        "batch": 512,
        "epochs": 20,
        "hidden": [256, 128],
        "dropout": 0.0,
        "init": "he",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 0,
    }
    mismatched = {
        key: (cfg.get(key), value)
        for key, value in expected.items()
        if cfg.get(key) != value
    }
    if mismatched:
        raise ValueError(f"CE result does not match the requested MSE setup: {mismatched}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    data = prepare_data(
        device,
        val_fraction=0.2,
        seed=42,
        processed_dir=str(args.processed_dir),
    )

    mse_cfg = {
        **cfg,
        "exp_id": "loss-mse-s0",
        "group": "loss",
        "description": "MSE loss, same baseline configuration, seed 0",
        "loss": "mse",
    }
    mse_result = run_experiment(mse_cfg, data)
    if mse_result["summary"]["diverged"]:
        raise RuntimeError("MSE run diverged; results were not saved.")

    figures_dir = args.output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    result_path = save_result(mse_result, str(results_dir))
    plot_run(mse_result, str(figures_dir / "loss-mse-s0.png"))
    plot_compare(
        [ce_result, mse_result],
        "val_acc",
        str(figures_dir / "compare_loss_val_acc.png"),
        "CE vs MSE — validation accuracy",
    )
    plot_compare(
        [ce_result, mse_result],
        "val_macro_f1",
        str(figures_dir / "compare_loss_val_macro_f1.png"),
        "CE vs MSE — validation macro-F1",
    )

    summary = mse_result["summary"]
    print(f"Saved MSE result: {result_path}")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Validation accuracy: {summary['val_acc']:.6f}")
    print(f"Validation macro-F1: {summary['val_macro_f1']:.6f}")


if __name__ == "__main__":
    main()
