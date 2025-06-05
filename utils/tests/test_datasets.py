import pytest
import pandas as pd
from utils.datasets import get_dataset


@pytest.mark.parametrize(
    "dataset",
    ["HotpotQA", "2WikiMultihopQA", "MuSiQue"],
)
def test_get_dataset(dataset):
    df = get_dataset(dataset, num_samples=5)
    assert isinstance(df, pd.DataFrame), "The returned object is not a DataFrame"
    assert set(df.columns) == {"id", "question", "answer"}, (
        "DataFrame columns do not match expected"
    )
    assert len(df) == 5, "The number of samples in the DataFrame is not equal to 5"


def test_get_invalid_dataset():
    with pytest.raises(ValueError):
        get_dataset("InvalidDatasetName")
