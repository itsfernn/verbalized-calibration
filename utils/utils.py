import re
import matplotlib.pyplot as plt
import numpy as np


# see https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#F1
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import re
    import string

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


def compute_exact_match(prediction, truth):
    """
    Compute Exact Match (EM) score between prediction and truth.
    EM is 1 if the prediction matches the truth exactly, otherwise 0.
    """
    return int(normalize_text(prediction) == normalize_text(truth))


def extract_answer_and_confidence(response: str):
    """
    Extracts the final answer and confidence/probability from the LLM response.
    Expected format at the end:
    "Final Answer: <answer> Probability: <confidence>"
    """
    response = response.strip().replace("\n", " ")

    # Regex to catch variations in punctuation/spacing
    pattern = r"(?:Final Answer|Answer|Guess)[:\s]*([^\n]+?)\s+(?:Probability|Confidence)[:\-\s]*([01](?:\.\d+)?)"

    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        confidence = float(match.group(2).strip())
        return answer, confidence

    # Fallback: try to extract confidence separately if answer parsing failed
    return extract_answer(response), extract_confidence(response)


def extract_answer(response: str):
    """
    Extracts just the final answer if confidence can't be found.
    """
    pattern = r"(?:Final Answer|Answer|Guess)[:\s]*([^\n]+)"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_confidence(response: str):
    """
    Extracts a floating point number labeled as a confidence/probability in the text.
    This acts as a fallback if the main extractor fails.
    """
    response = response.replace("\n", " ").strip()

    # More general pattern to catch "Probability: 0.92", "Confidence - 0.8", etc.
    pattern = r"(?:probability|confidence)[:\-\s]+([01](?:\.\d+)?)"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return float(match.group(1).strip())
    return None


# adapted from https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d/
def expected_calibration_error(scores, confidences, M=10):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower &amp; upper)
        in_bin = np.logical_and(
            confidences > bin_lower.item(), confidences <= bin_upper.item()
        )
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = scores[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece


def plot_confidence_error(
    scores,
    confidences,
    M=10,
    xlabel="Confidence",
    ylabel="Accuracy",
    title="Calibration Plot",
):
    """
    Plots a reliability diagram comparing confidence to accuracy.

    Parameters:
        scores (np.array): Binary correctness of predictions (0 or 1).
        confidences (np.array): Model confidence scores (0 to 1).
        M (int): Number of bins.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        title (str): Plot title.

    Returns:
        Matplotlib figure and axis.
    """

    # Define bin edges and centers
    bin_edges = np.linspace(0, 1, M + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute bin indices
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, M - 1)  # Ensure indices stay within range

    # Compute bin sizes and accuracy per bin
    bin_sizes = np.bincount(bin_indices, minlength=M)
    accuracy_per_bin = np.zeros(M)

    for i in range(M):
        if bin_sizes[i] > 0:
            accuracy_per_bin[i] = scores[bin_indices == i].mean()

    # Normalize bin sizes for color scaling
    max_bin_size = max(bin_sizes) if max(bin_sizes) > 0 else 1
    color_intensity = bin_sizes / max_bin_size  # Normalize between 0 and 1
    color_map = plt.get_cmap("PuBu")

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot bars with color gradient
    ax.bar(
        bin_centers,
        accuracy_per_bin,
        width=0.08,
        color=color_map(color_intensity),
        edgecolor="black",
    )

    # Plot identity line
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Perfect Calibration",
    )

    # Add Expected Calibration Error (ECE)
    ece_value = expected_calibration_error(scores, confidences, M)
    ax.text(
        0.1,
        0.9,
        f"ECE: {ece_value:.4f}",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )

    # Labels and styling
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(1.0)
    ax.set_ylim(1.0)
    ax.grid(True, linestyle="--", alpha=0.7)

    return fig, ax
