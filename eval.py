import argparse
import wandb
import sklearn.metrics
from utils.utils import expected_calibration_error, plot_confidence_error


def main(args=None):
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")
    parser.add_argument(
        "--table",
        type=str,
        required=True,
        help="wandb artifact table to responses",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="uncertainty-reimplimentation-results",
        help="Name of the project",
    )
    args = parser.parse_args(args)
    print(args)

    run = wandb.init(project=args.project_name, job_type="eval", config=args)
    table_artifact = run.use_artifact(f"{args.project_name}/{args.table}:latest")
    table = table_artifact.get("table").get_dataframe()
    run.config.update(table_artifact.logged_by().config)

    xlabel = f"{args.aggregation} " if args.aggregation else ""
    xlabel += f"{run.config.confidence_strategy} "
    xlabel += "confidence"

    fig, _ = plot_confidence_error(
        table[args.score],
        table["em"],
        title=f"{run.config.dataset_name} {run.config.model}",
        ylabel=args.score.title(),
        xlabel=xlabel.title(),
    )

    wandb.log(
        {
            "f1": table["f1"].mean(),
            "acc": table["em"].mean(),
            "ece": expected_calibration_error(table[args.score], table["em"]),
            "auroc": calculate_auroc(table, args.score),
            "macro_ece": calculate_marco_ece(table, args.score),
        }
    )

    wandb.log({"calibration_plot": wandb.Image(fig)})
    # TODO log table again if it changed

    run.finish()


def calculate_marco_ece(table, eval_column):
    table_p = table[table["em"]]
    ice_p = expected_calibration_error(table_p[eval_column], table_p["em"])

    table_n = table[table["em"] == 0]
    ice_n = expected_calibration_error(table_n[eval_column], table_n["em"])

    macro_ece = (ice_p + ice_n) / 2

    return macro_ece


def calculate_auroc(table, eval_column):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        y_true=table[eval_column], y_score=table["em"], pos_label=1
    )
    auroc = sklearn.metrics.auc(fpr, tpr)
    return auroc


if __name__ == "__main__":
    main()
