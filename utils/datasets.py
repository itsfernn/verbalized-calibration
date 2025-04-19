from datasets import load_dataset, Dataset
import json
import pandas as pd
from utils.utils import normalize_text


def get_dataset(dataset_name, num_samples=500) -> pd.DataFrame:
    if "hotpotqa" == normalize_text(dataset_name):
        dataset = load_dataset(
            "hotpotqa/hotpot_qa", "distractor", split=f"validation[:{num_samples}]"
        )
        assert isinstance(dataset, Dataset)
        df = dataset.to_pandas()

    elif "2wiki" in normalize_text(dataset_name):
        with open("datasets/2WikiMultihopQA/dev.json", "r") as f:
            data = json.loads(f.read())

        df = pd.DataFrame(data[:num_samples])
        df["id"] = df["_id"]

    elif "musique" == normalize_text(dataset_name):
        dataset = load_dataset(
            "bdsaglam/musique", "answerable", split=f"validation[:{num_samples}]"
        )
        assert isinstance(dataset, Dataset)
        df = dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)
        df["id"] = df["_id"]
    else:
        raise ValueError()

    assert isinstance(df, pd.DataFrame)
    df = df[["id", "question", "answer"]]
    assert isinstance(df, pd.DataFrame)
    return df
