import pytest
from utils.processor import (
    DirectProcessor,
    TopKProcessor,
    CotProcessor,
    MultistepProcessor,
    MultistepBeamSearchProcessor,
)


@pytest.fixture
def sample_data():
    return {"question": "What is the capital of France?", "answer": "Paris"}


def test_direct_processor(sample_data):
    processor = DirectProcessor(llm_config={})
    result = processor.process_sample(sample_data)
    assert result["prediction"] is not None
    assert 0.0 <= result["confidence"] <= 1.0


def test_top_k_processor(sample_data):
    processor = TopKProcessor(llm_config={}, k=3)
    result = processor.process_sample(sample_data)
    assert result["prediction"] is not None
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0


def test_cot_processor(sample_data):
    processor = CotProcessor(llm_config={})
    result = processor.process_sample(sample_data)
    assert result["prediction"] is not None
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0


def test_multistep_processor(sample_data):
    processor = MultistepProcessor(llm_config={}, k=3)
    result = processor.process_sample(sample_data)
    assert result["prediction"] is not None
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0


@pytest.mark.parametrize(
    "search_criterion",
    [
        "highest_confidence",
        "median_confidence",
        "highest_aggregated_confidence",
        "median_aggregated_confidence",
    ],
)
def test_multistep_beam_search_processor(sample_data, search_criterion):
    processor = MultistepBeamSearchProcessor(
        k=3,
        llm_config={},
        branching_factor=2,
        beam_width=2,
        search_criterion=search_criterion,
    )
    result = processor.process_sample(sample_data)
    assert result["prediction"] is not None
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0
