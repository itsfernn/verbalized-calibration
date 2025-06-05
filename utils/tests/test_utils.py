import pandas as pd
from utils.utils import calculate_macro_ece

import pytest
import numpy as np
from utils.utils import (
    normalize_text,
    compute_f1,
    compute_exact_match,
    extract_texts_and_confidences,
    extract_answer,
    expected_calibration_error,
    plot_confidence_error,
)


def test_normalize_text():
    assert normalize_text("The quick brown fox.") == "quick brown fox"
    assert normalize_text("A test, with punctuation!") == "test with punctuation"
    assert normalize_text("An example.") == "example"


def test_compute_f1():
    assert compute_f1("The quick brown fox", "The quick brown fox") == 1
    assert compute_f1("The quick brown fox", "The brown fox") < 1
    assert compute_f1("The quick brown fox", "The brown fox") > 0
    assert compute_f1("Apple", "Banana") == 0
    assert compute_f1("The quick brown fox", "") == 0


def test_compute_exact_match():
    assert compute_exact_match("The quick brown fox", "The quick brown fox") == 1
    assert compute_exact_match("The quick brown fox", "The brown fox") == 0


def test_extract_texts_and_confidences():
    response = "Final Answer: The quick brown fox Probability: 0.95"
    # assert extract_texts_and_confidences(response) == [("The quick brown fox", 0.95)]

    response = "Answer: No. Confidence: 0.9 Answer: Yes. Confidence: 0.1 Answer: Different nationalities. Confidence: 0.8 Answer: American and American. Confidence: 0.3 Answer: Both American. Confidence: 0.5"
    matches = extract_texts_and_confidences(response)
    assert len(matches) == 5
    assert matches[0][1] == 0.9

    with pytest.raises(ValueError):
        extract_texts_and_confidences("Invalid response format")


def test_extract_answer():
    response = "Final Answer: The quick brown fox Confidence: 0.95"
    assert extract_answer(response) == "The quick brown fox"
    assert extract_answer("No answer provided.") is None


def test_expected_calibration_error():
    scores = np.array([1, 0, 1, 1])
    confidences = np.array([0.9, 0.1, 0.8, 0.7])
    ece = expected_calibration_error(scores, confidences, M=2)
    assert isinstance(ece, float)


def test_plot_confidence_error():
    scores = np.array([1, 0, 1, 1])
    confidences = np.array([0.9, 0.1, 0.8, 0.7])
    fig, ax = plot_confidence_error(scores, confidences, M=2)
    assert fig is not None
    assert ax is not None


def test_calculate_marco_ece():
    # Test case 1: Perfect calibration
    table_perfect = pd.DataFrame(
        {"em": [0, 0, 1, 0], "confidence": [0.0, 0.0, 1.0, 0.0]}
    )
    calibration_error_perfect = calculate_macro_ece(table_perfect)
    assert calibration_error_perfect == 0.0

    # Test case 2: Worst calibration
    table_worst = pd.DataFrame({"em": [1, 1, 0, 1], "confidence": [0.0, 0.0, 1.0, 0.0]})
    calibration_error_worst = calculate_macro_ece(table_worst)
    assert calibration_error_worst == 1.0

    # Test case 2: Worst calibration
    table_worst = pd.DataFrame({"em": [0, 0, 0, 0], "confidence": [0.0, 0.0, 0.0, 0.0]})
    calibration_error_worst = calculate_macro_ece(table_worst)
    assert calibration_error_worst == 0.0

    # Test case 3: Imperfect calibration
    table_imperfect1 = pd.DataFrame(
        {"em": [1, 1, 0, 0], "confidence": [0.9, 0.8, 0.1, 0.2]}
    )
    calibration_error_imperfect1 = calculate_macro_ece(table_imperfect1)

    # Test case 4: Imperfect calibration
    table_imperfect2 = pd.DataFrame(
        {"em": [1, 0, 1, 0], "confidence": [0.9, 0.9, 0.1, 0.1]}
    )
    calibration_error_imperfect2 = calculate_macro_ece(table_imperfect2)

    assert calibration_error_imperfect1 < calibration_error_imperfect2

    # Test case 5: Perfect calibration (repeated for completeness)
    table_perfect_repeated = pd.DataFrame(
        {"em": [0, 0, 1, 0], "confidence": [0.0, 0.0, 1.0, 0.0]}
    )
    calibration_error_perfect_repeated = calculate_macro_ece(table_perfect_repeated)
    assert calibration_error_perfect_repeated == 0.0
