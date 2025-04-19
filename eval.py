import argparse
import wandb
import math
import sklearn.metrics
import concurrent.futures
from tqdm import tqdm
from utils import extract_probability, expected_calibration_error, plot_confidence_error, compute_exact_match, compute_f1

def main(args=None):
    parser = argparse.ArgumentParser(description='Evaluate a model on a dataset.')
    parser.add_argument('--score', type=str, default="em", help='Type of score to use (f1, gpt_eval,...)')
    parser.add_argument('--aggregation', help='How to aggregate multiple confidence values (multiplied, final, max, min, average)')
    parser.add_argument('--table', type=str, required=True, help='wandb artifact table to responses')
    parser.add_argument('--recompute', action='store_true', help='Recompute the scores')
    parser.add_argument('--no-logging', action='store_true', help='Do not log the results')
    parser.add_argument('--project_name', type=str, default='calibration_paper', help='Name of the project')
    args = parser.parse_args(args)
    print(args)

    if args.no_logging:
        # TODO disable wandb
        pass
    else:
        run = wandb.init(project=args.project_name, job_type='eval', config=args)
        table_artifact = run.use_artifact(f"{args.project_name}/{args.table}:latest")
        table = table_artifact.get("table").get_dataframe()
        run.config.update(table_artifact.logged_by().config)

        ## TODO move this to processor
        if args.aggregation and (args.aggregation not in table.columns or args.recompute):
            table[args.aggregation] = table.apply(lambda row: compute_confidence(row, args.aggregation), axis=1)

        if args.score not in table.columns or args.recompute:
            if args.score == "gpt_eval":
                with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
                    table["gpt_eval"] = list(tqdm(executor.map(lambda row: gpt_eval(row["question"], row["answer"], row["prediction"]), table.to_dict("records")), total=table.shape[0]))
                wandb.log({"table" : wandb.Table(dataframe=table)})
            elif args.score == "em":
                table["em"] = table.apply(lambda row: compute_exact_match(row["prediction"], row["answer"]), axis=1)
            elif args.score == "f1":
                table["f1"] = table.apply(lambda row: compute_f1(row["prediction"], row["answer"]), axis=1)
            else:
                raise NotImplementedError(f"Recomputing {args.score} is not implemented yet")
            # TODO recompute f1

        confidence_column = "probability" if args.aggregation is None else args.aggregation

        xlabel = f"{args.aggregation} " if args.aggregation else ''
        xlabel += f"{run.config.confidence_strategy} "
        xlabel += "confidence"

        fig, ax = plot_confidence_error(
            table[args.score],
            table[confidence_column],
            title=f"{run.config.dataset_name} {run.config.model}",
            ylabel=args.score.title(),
            xlabel=xlabel.title(),
        )

        wandb.log({
                "f1": table["f1"].mean(),
                "acc": table["em"].mean(),
                "ece" : expected_calibration_error(table[args.score], table[confidence_column]),
                "auroc" : calculate_auroc(table, args.score, confidence_column),
                "macro_ece" : calculate_marco_ece(table, args.score, confidence_column),
            })

        wandb.log({"calibration_plot" : wandb.Image(fig)})
        # TODO log table again if it changed

    run.finish()



def compute_confidence(row, method: str):
    reasoning = row["reasoning"]
    reasoning_steps = reasoning.split("\n")[:-1]
    reasoning_steps = [ step for step in reasoning_steps if step ]  # remove empty steps

    if not reasoning_steps: # no reasoning
        return 0.5

    probs = list(map(extract_probability, reasoning_steps))

    if method == "multiplied": 
        return math.prod(probs)
    elif method == "final":
        return extract_probability(reasoning_steps[-1])
    elif method == "average":
        return sum(probs) / len(probs)
    elif method == "power":
        return extract_probability(reasoning_steps[-1]) ** len(reasoning_steps)
    elif method == "max":
        return max(probs)
    elif method == "min":
        return min(probs)
    else:
        raise NotImplementedError(f"Confidence method {method} not implemented")


from chat import Chat
from utils import normalize_text
def gpt_eval(question, answer, prediction):
    """
    Evaluate the answer and prediction using GPT-4o-mini.
    """
    if normalize_text(answer) == normalize_text(prediction):
        return True
    chat = Chat()
    prompt = (
        "Are the following two answers to my question Q semantically equivalent?\n\n"
        f"Q: {question}\n"
        f"A1: {answer}\n"
        f"A2: {prediction}\n\n"
        "Please answer with a single word, either \"Yes\" or \"No\""
    )
    chat.add_message(prompt, "user")
    response = chat.response()
    response = normalize_text(response)
    return "yes" in response


def calculate_marco_ece(table, eval_column, confidence_column):
    table_p = table[table["em"]]
    ice_p = expected_calibration_error(table_p[eval_column], table_p[confidence_column])

    table_n = table[table["em"] == 0]
    ice_n = expected_calibration_error(table_n[eval_column], table_n[confidence_column])

    macro_ece = (ice_p + ice_n) / 2

    return macro_ece

def calculate_auroc(table, eval_column, confidence_column):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true = table[eval_column], y_score = table[confidence_column], pos_label = 1)
    auroc = sklearn.metrics.auc(fpr, tpr)
    return auroc


if __name__ == '__main__':
    main()

