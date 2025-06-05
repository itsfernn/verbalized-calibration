import re
import matplotlib.pyplot as plt
import sklearn
import numpy as np


# see https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#F1
def normalize_text(s: str):
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


def compute_f1(prediction: str, truth: str):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # Handle no-answer cases
    if not pred_tokens or not truth_tokens:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # Return 0 if there are no common tokens
    if not common_tokens:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


def compute_exact_match(prediction: str, truth: str) -> int:
    """
    Compute Exact Match (EM) score between prediction and truth.
    EM is 1 if the prediction matches the truth exactly, otherwise 0.
    """
    normalized_prediction = normalize_text(prediction)
    normalized_truth = normalize_text(truth)
    return int(normalized_prediction == normalized_truth)


def calculate_macro_ece(table, collumn="em", give_array=False):
    classes = [0, 1]
    table_c = [table[table[collumn] == c] for c in classes]
    table_c = [t for t in table_c if not t.empty]

    ece_c = [expected_calibration_error(t[collumn], t["confidence"]) for t in table_c]

    if give_array:
        return ece_c

    macro_ece = sum(ece_c) / len(ece_c)
    
    if give_array:
        return ece_c

    return macro_ece


def calculate_auroc(exact_match, confidence):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        y_true=exact_match, y_score=confidence, pos_label=1
    )
    auroc = sklearn.metrics.auc(fpr, tpr)
    return auroc


def extract_texts_and_confidences(response: str) -> tuple[list[str], list[float]]:
    """
    Extracts the final answer and confidence/probability from the LLM response.
    Expected format at the end:
    "Final Answer: <answer> Probability: <confidence>"
    """
    response = response.strip().replace("\n", " ")

    # Regex to catch variations in punctuation/spacing
    pattern = r"(?:Final Answer|Answer|Guess|Step \d*)[:\-\s]*(.*?)\s*(?:Probability|Confidence)[:\-\s]*([01](?:\.\d+)?)"

    matches = re.findall(pattern, response, flags=re.MULTILINE)
    texts = [text for text, _ in matches]
    confidences = [float(confidence) for _, confidence in matches]

    return texts, confidences


def extract_answer(response: str) -> str | None:
    pattern = r"(?:Final Answer|Answer|Guess)[:\s]*([^\n]+)(?:Confidence|Probability)"
    match = re.search(pattern, response)
    return match.group(1).strip() if match else None


# adapted from https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d/
def expected_calibration_error(scores, confidences, M=10):
    bin_boundaries = np.linspace(0.0, 1.0, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for i in range(M):
        bin_lower = bin_lowers[i]
        bin_upper = bin_uppers[i]

        # Include lower bound and exclude upper bound, except for last bin
        if i == M - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        if np.any(in_bin):
            prob_in_bin = np.mean(in_bin)
            accuracy_in_bin = scores[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

    return ece


def plot_confidence_error(
    scores,
    confidences,
    ax=None,
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
    fig = None
    if ax is None:
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
        0.05,
        0.88,
        f"ECE: {ece_value:.4f}",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"),
    )

    # Labels and styling
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, 1])  # type: ignore
    ax.set_ylim([0, 1])  # type: ignore
    #ax.grid(True, linestyle="--", alpha=0.7)

    if fig == None:
        return ax
    else:
        return fig, ax


def plot_confidence_error_old(
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
