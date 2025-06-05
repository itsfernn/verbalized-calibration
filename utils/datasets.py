from datasets import load_dataset, Dataset
import pandas as pd


def get_dataset(dataset, num_samples=500, seed=42, all_collumns=False) -> Dataset:
    if dataset == "HotpotQA":
        ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train")
        assert isinstance(ds, Dataset)
        ds = ds.shuffle(seed=seed).select(range(num_samples))

        if not all_collumns:
            ds = ds.select_columns(["id", "question", "answer", "type", "level"])

    elif dataset == "2WikiMultihopQA":
        path = "datasets/2WikiMultihopQA/"
        df = pd.read_json(path + "dev.jsonl", orient="records", lines=True)
        ds = Dataset.from_pandas(df)
        ds = ds.rename_column("_id", "id")
        ds = ds.remove_columns(["evidences"])
        ds = ds.shuffle(seed=seed).select(range(num_samples))

        if not all_collumns:
            ds = ds.select_columns(["id", "question", "answer", "type"])

    elif dataset == "MuSiQue":
        ds = load_dataset("bdsaglam/musique", "answerable", split="validation")
        assert isinstance(ds, Dataset)
        ds = ds.shuffle(seed=seed).select(range(num_samples))
        ds = ds.remove_columns(
            ["question_decomposition", "answer_aliases", "answerable"]
        )

        if not all_collumns:
            ds = ds.select_columns(["id", "question", "answer"])

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return ds
