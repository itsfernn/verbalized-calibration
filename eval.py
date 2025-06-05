import argparse
import pandas as pd
from utils.processor import get_processor
from utils.datasets import get_dataset
from utils.utils import (
    plot_confidence_error,
    expected_calibration_error,
    calculate_auroc,
    calculate_macro_ece,
)
import wandb

import os


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Evaluate results from a previous inference run."
    )
    parser.add_argument(
        "--results_table",
        required=True,
        help="The ID of the Weights & Biases table artifact containing inference results.",
    )
    parser.add_argument(
        "--project_name",
        default="uncertainty-reimplimentation-results",
        help="Name of the Weights & Biases project where the results table is located.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for evaluation.",
    )
    parser.add_argument(
        "--eval",
        default="em",
        help="Type of eval to use to calculate metrics (em, gpt-eval)"
    )
    parser.add_argument(
        "--job_type",
        default="eval",
        help="Change Job type"
    )
    args = parser.parse_args(args)
    run = wandb.init(
        project=args.project_name,
        job_type=args.job_type if not args.debug else "debug-eval",
        config=args,  # type: ignore
    )

    table_artifact = run.use_artifact(
        f"{args.project_name}/{args.results_table}:latest"
    )
    table: pd.DataFrame = table_artifact.get("table").get_dataframe()
    run.config.update(table_artifact.logged_by().config)

    fig, _ = plot_confidence_error(
        table[args.eval],
        table["confidence"],
        title=f"{run.config.prompt} ({run.config.dataset}, {run.config.model})",
        ylabel="Accuracy",
        xlabel="Verbalized Confidence",
    )

    table = table.dropna()  ## remove missing rows

    f1_score = table["f1"].mean()
    accuracy = table[args.eval].mean()
    ece_score = expected_calibration_error(table[args.eval], table["confidence"])
    auroc_score = calculate_auroc(table[args.eval], table["confidence"])
    macro_ece_score = calculate_macro_ece(table, collumn=args.eval)

    wandb.log(
        {
            "f1": f1_score,
            "acc": accuracy,
            "ece": ece_score,
            "auroc": auroc_score,
            "macro_ece": macro_ece_score,
        }
    )

    wandb.log({"calibration_plot": wandb.Image(fig)})

    run.finish()



if __name__ == "__main__":
    main()
