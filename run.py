import argparse
import concurrent.futures
import pandas as pd
from tqdm import tqdm
from utils.processor import get_processor
from utils.datasets import get_dataset
from bert_score import score
import wandb
import eval


# Main function to run the process
def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Run inference and evaluation for LLM model."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--prompt_strategy",
        type=str,
        required=True,
        help="Prompt strategy (e.g., cot or direct)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples",
    )
    parser.add_argument(
        "--confidence_strategy",
        type=str,
        default="verbalized",
        required=True,
        help="Confidence score method (so far only verb)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature of Model",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use for parallel processing",
    )
    parser.add_argument(
        "--project_name",
        default="uncertainty-reimplimentation-results",
        help="WandB project name",
    )

    args = parser.parse_args()
    args.method_name = f"{args.prompt_strategy}_{args.confidence_strategy}"
    print(args)

    run = wandb.init(
        project=args.project_name,
        job_type="inference",
        config=args,
    )

    # Load LLM config
    llm_config = {
        "model": args.model,
        "temperature": args.temperature,
    }

    # Load Dataframe form disk
    dataset = get_dataset(args.dataset_name, args.num_samples)

    # Get the correct processor using factory pattern
    Processor_CLS = get_processor(args.method_name)
    processor = Processor_CLS(llm_config)

    # Process the dataset using the processor in parallel
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        responses = pd.DataFrame(
            tqdm(
                executor.map(processor, dataset.to_dict("records")),
                total=dataset.shape[0],
            )
        )

    # Combine dataset with responses
    df = pd.concat([dataset, responses], axis=1)

    # Get BERT score
    _, _, df["bert_score"] = score(
        df["answer"], df["prediction"], lang="en", rescale_with_baseline=True
    )

    # Log Table
    table = wandb.Table(dataframe=df)
    wandb.log({"table": table})
    wandb.finish()

    results_table = f"run-{run.id}-table"

    # TODO: eval
    eval.main(["--table", output_file_path])


# Entry point for script
if __name__ == "__main__":
    main()
