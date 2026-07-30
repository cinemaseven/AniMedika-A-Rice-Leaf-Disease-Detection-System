from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import EXPERIMENT_OUTPUTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="*")
    args = parser.parse_args()

    names = args.experiments or [path.name for path in EXPERIMENT_OUTPUTS_DIR.iterdir() if path.is_dir()]
    rows = []
    for name in names:
        fold_path = EXPERIMENT_OUTPUTS_DIR / name / "kfold" / "kfold_fold_results.csv"
        if not fold_path.exists():
            continue
        frame = pd.read_csv(fold_path)
        rows.append(
            {
                "experiment": name,
                "validation_accuracy_mean": frame["validation_accuracy"].mean(),
                "validation_accuracy_std": frame["validation_accuracy"].std(),
                "validation_f1_macro_mean": frame["validation_f1_macro"].mean(),
                "validation_f1_macro_std": frame["validation_f1_macro"].std(),
                "validation_log_loss_mean": frame["validation_log_loss"].mean(),
                "validation_roc_auc_macro_mean": frame["validation_roc_auc_ovr_macro"].mean(),
                "accuracy_gap_mean": frame["gap_accuracy"].mean(),
                "f1_gap_mean": frame["gap_f1_macro"].mean(),
            }
        )

    if not rows:
        raise SystemExit("No completed k-fold experiment results were found.")
    leaderboard = pd.DataFrame(rows).sort_values(
        ["validation_f1_macro_mean", "validation_accuracy_mean", "validation_log_loss_mean"],
        ascending=[False, False, True],
    )
    save_path = EXPERIMENT_OUTPUTS_DIR / "experiment_leaderboard.csv"
    leaderboard.to_csv(save_path, index=False)
    print(leaderboard.to_string(index=False))
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
