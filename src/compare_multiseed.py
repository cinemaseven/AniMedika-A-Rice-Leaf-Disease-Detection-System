from __future__ import annotations

import argparse

import pandas as pd

from config import EXPERIMENT_OUTPUTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare configurations across multiple random seeds."
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Base experiment names to compare.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Random seeds used for every experiment.",
    )
    args = parser.parse_args()

    run_rows: list[dict] = []

    for experiment_name in args.experiments:
        for seed in args.seeds:
            run_name = f"{experiment_name}_seed{seed}"
            kfold_dir = (
                EXPERIMENT_OUTPUTS_DIR
                / run_name
                / "kfold"
            )

            fold_path = kfold_dir / "kfold_fold_results.csv"
            class_path = kfold_dir / "oof_per_class_metrics.csv"
            prediction_path = kfold_dir / "oof_predictions.csv"

            missing = [
                path
                for path in [fold_path, class_path, prediction_path]
                if not path.exists()
            ]

            if missing:
                missing_text = "\n".join(str(path) for path in missing)
                raise FileNotFoundError(
                    f"Missing result files for {run_name}:\n{missing_text}"
                )

            fold_results = pd.read_csv(fold_path)
            class_results = pd.read_csv(class_path)
            predictions = pd.read_csv(prediction_path)

            rice_blast = class_results.loc[
                class_results["class"] == "Rice_Blast"
            ]
            sheath_blight = class_results.loc[
                class_results["class"] == "Sheath_Blight"
            ]

            if rice_blast.empty or sheath_blight.empty:
                raise ValueError(
                    f"Required class metrics were not found for {run_name}"
                )

            rice_blast = rice_blast.iloc[0]
            sheath_blight = sheath_blight.iloc[0]

            blast_to_sheath = (
                (predictions["actual_class"] == "Rice_Blast")
                & (predictions["predicted_class"] == "Sheath_Blight")
            ).sum()

            sheath_to_blast = (
                (predictions["actual_class"] == "Sheath_Blight")
                & (predictions["predicted_class"] == "Rice_Blast")
            ).sum()

            run_rows.append(
                {
                    "experiment": experiment_name,
                    "seed": seed,
                    "validation_accuracy_mean": (
                        fold_results["validation_accuracy"].mean()
                    ),
                    "validation_f1_macro_mean": (
                        fold_results["validation_f1_macro"].mean()
                    ),
                    "validation_log_loss_mean": (
                        fold_results["validation_log_loss"].mean()
                    ),
                    "validation_roc_auc_macro_mean": (
                        fold_results[
                            "validation_roc_auc_ovr_macro"
                        ].mean()
                    ),
                    "accuracy_gap_mean": (
                        fold_results["gap_accuracy"].mean()
                    ),
                    "f1_gap_mean": (
                        fold_results["gap_f1_macro"].mean()
                    ),
                    "rice_blast_recall": (
                        rice_blast["recall_sensitivity"]
                    ),
                    "rice_blast_f1": rice_blast["f1_score"],
                    "sheath_blight_f1": sheath_blight["f1_score"],
                    "blast_to_sheath_errors": int(blast_to_sheath),
                    "sheath_to_blast_errors": int(sheath_to_blast),
                    "blast_sheath_total_errors": int(
                        blast_to_sheath + sheath_to_blast
                    ),
                }
            )

    per_seed = pd.DataFrame(run_rows)

    summary = (
        per_seed.groupby("experiment", as_index=False)
        .agg(
            seed_count=("seed", "count"),
            accuracy_mean_across_seeds=(
                "validation_accuracy_mean",
                "mean",
            ),
            accuracy_std_across_seeds=(
                "validation_accuracy_mean",
                "std",
            ),
            macro_f1_mean_across_seeds=(
                "validation_f1_macro_mean",
                "mean",
            ),
            macro_f1_std_across_seeds=(
                "validation_f1_macro_mean",
                "std",
            ),
            log_loss_mean_across_seeds=(
                "validation_log_loss_mean",
                "mean",
            ),
            roc_auc_mean_across_seeds=(
                "validation_roc_auc_macro_mean",
                "mean",
            ),
            accuracy_gap_mean_across_seeds=(
                "accuracy_gap_mean",
                "mean",
            ),
            f1_gap_mean_across_seeds=(
                "f1_gap_mean",
                "mean",
            ),
            rice_blast_recall_mean=(
                "rice_blast_recall",
                "mean",
            ),
            rice_blast_f1_mean=(
                "rice_blast_f1",
                "mean",
            ),
            sheath_blight_f1_mean=(
                "sheath_blight_f1",
                "mean",
            ),
            blast_sheath_errors_mean=(
                "blast_sheath_total_errors",
                "mean",
            ),
        )
        .sort_values(
            [
                "macro_f1_mean_across_seeds",
                "accuracy_mean_across_seeds",
                "log_loss_mean_across_seeds",
            ],
            ascending=[False, False, True],
        )
    )

    per_seed_path = (
        EXPERIMENT_OUTPUTS_DIR / "multiseed_per_seed_results.csv"
    )
    summary_path = (
        EXPERIMENT_OUTPUTS_DIR / "multiseed_summary.csv"
    )

    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nPER-SEED RESULTS")
    print(per_seed.to_string(index=False))

    print("\nMULTI-SEED SUMMARY")
    print(summary.to_string(index=False))

    print(f"\nSaved: {per_seed_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()