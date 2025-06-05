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

os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"


# Main function to run the process
def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Run inference and evaluation for LLM model."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to use (e.g., 'HotpotQA', '2WikiMultihopQA').",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the language model to use (e.g., 'gpt-4o-mini', 'deepseek-coder').",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompting strategy to apply ('direct', 'cot', 'topk', 'multistep').",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to process from the dataset.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature setting for the language model.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of parallel workers for processing samples.",
    )
    parser.add_argument(
        "--project_name",
        default="uncertainty-reimplimentation-results",
        help="Name of the Weights & Biases project to log results.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode, processing a smaller subset and logging to a debug job type.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="List of tags",
    )

    args = parser.parse_args()
    print(args)

    run = wandb.init(
        project=args.project_name,
        job_type="inference" if not args.debug else "debug",
        config=args,  # type: ignore
        tags=args.tags,
    )

    # Load Dataframe form disk
    dataset = get_dataset(args.dataset, args.num_samples)

    # Load LLM config
    llm_config = {
        "model": args.model,
        "temperature": args.temperature,
    }

    # Get the correct processor using factory pattern
    processor = get_processor(args.prompt, llm_config)

    # Process the dataset using the processor in parallel
    table = dataset.map(
        processor, num_proc=args.num_workers, load_from_cache_file=False
    )
    results = table.filter(lambda x: bool(x["texts"]), load_from_cache_file=False)
    results = results.map(processor.eval_sample, load_from_cache_file=False)
    results = results.remove_columns(["texts", "confidences"])

    failed = table.filter(lambda x: not bool(x["texts"]), load_from_cache_file=False)
    failed = failed.remove_columns(["texts", "confidences"])

    table = wandb.Table(dataframe=results.to_pandas())
    failed = wandb.Table(dataframe=failed.to_pandas())
    wandb.log({"table": table, "failed": failed})

    wandb.finish()

    results_table = f"run-{run.id}-table"
    args.results_table = results_table

    args_list = ["--results_table", results_table, "--project_name", args.project_name]

    if args.debug:
        args_list.append("--debug")

    eval(args_list)


def eval(args):
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
        help="Type of eval to use to calculate metrics (em, gpt-eval)",
    )
    parser.add_argument("--job_type", default="eval", help="Change Job type")
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
    macro_ece_score = calculate_macro_ece(table, args.eval)

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


# Entry point for script
if __name__ == "__main__":
    main()
