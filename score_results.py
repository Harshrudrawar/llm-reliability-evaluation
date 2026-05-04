import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd


# ============================================================
# Config
# ============================================================
RESULTS_DIR = Path("results")
RAW_OUTPUT_FILE = RESULTS_DIR / "raw_outputs.csv"
SCORES_FILE = RESULTS_DIR / "scores.csv"
SUMMARY_JSON = RESULTS_DIR / "summary.json"

# These weights should match your methodology document
WEIGHTS = {
    "consistency": 0.15,
    "robustness": 0.15,
    "reasoning_stability": 0.20,
    "long_context": 0.15,
    "edge_case": 0.10,
    "response_variance": 0.10,
    "parameter_sensitivity": 0.15,
}


# ============================================================
# Text helpers
# ============================================================
def normalize(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def pairwise_mean_similarity(texts) -> float:
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) < 2:
        return 1.0 if len(texts) == 1 else 0.0

    scores = []
    for a, b in combinations(texts, 2):
        scores.append(similarity(a, b))
    return sum(scores) / len(scores) if scores else 0.0


def safe_mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def pct(x: float) -> float:
    return round(x * 100.0, 2)


# ============================================================
# Category scorers
# ============================================================
def score_consistency(df: pd.DataFrame) -> float:
    """
    Average pairwise similarity across repeated outputs for each prompt.
    """
    scores = []
    for _, grp in df.groupby(["prompt_index"]):
        outputs = grp["output"].dropna().tolist()
        if len(outputs) >= 2:
            scores.append(pairwise_mean_similarity(outputs))
    return safe_mean(scores)


def score_robustness(df: pd.DataFrame) -> float:
    """
    Average similarity across paraphrased prompts in the robustness set.
    """
    outputs = df["output"].dropna().tolist()
    return pairwise_mean_similarity(outputs)


def classify_diamond_answer(text: str) -> str:
    """
    Classify reasoning-stability outputs into coarse labels.
    """
    t = normalize(text)

    if "cup" in t and "bed" in t:
        return "ambiguous"
    if "cup" in t:
        return "cup"
    if "bed" in t:
        return "bed"
    if "fridge" in t and "cup" not in t and "bed" not in t:
        return "other"
    return "other"


def score_reasoning_stability(df: pd.DataFrame) -> float:
    """
    Use the most frequent coarse answer label across all runs/challenge responses.
    """
    scores = []

    for _, grp in df.groupby(["prompt_index"]):
        labels = [classify_diamond_answer(o) for o in grp["output"].dropna().tolist()]
        labels = [l for l in labels if l != "other"]

        if not labels:
            scores.append(0.0)
            continue

        counts = Counter(labels)
        most_common_count = counts.most_common(1)[0][1]
        scores.append(most_common_count / len(labels))

    return safe_mean(scores)


def score_long_context(df: pd.DataFrame) -> float:
    """
    Task-specific scoring based on your exact long-context questions.
    Returns proportion correct.
    """
    total = 0
    correct = 0

    for _, row in df.iterrows():
        question = normalize(row.get("question", ""))
        output = normalize(row.get("output", ""))

        if not question:
            continue

        total += 1

        # Q1: title / main idea
        if "best title" in question:
            # Accept answers that capture the title meaning
            if ("great art" in output and "business as usual" in output) or (
                "great art" in output and "business" in output
            ):
                correct += 1

        # Q2: where was Portrait of Dr. Gachet exhibited
        elif "portrait of dr gachet" in question or ("where was" in question and "disappeared" in question):
            if "metropolitan museum of modern art" in output or "met museum" in output:
                correct += 1

        # Q3: main argument
        elif "main argument" in question:
            if (
                "commercial" in output
                or "big business" in output
                or "market value" in output
                or "commodity" in output
            ):
                correct += 1

    return (correct / total) if total else 0.0


def score_edge_case(df: pd.DataFrame) -> float:
    """
    Score whether the model correctly acknowledges uncertainty for impossible/future queries.
    """
    uncertainty_markers = [
        "not been announced",
        "not yet announced",
        "has not been announced",
        "not yet occurred",
        "cannot know",
        "no official",
        "not available",
        "has not been awarded",
        "has not happened yet",
    ]

    total = 0
    correct = 0

    for _, row in df.iterrows():
        output = normalize(row.get("output", ""))
        total += 1
        if any(marker in output for marker in uncertainty_markers):
            correct += 1

    return (correct / total) if total else 0.0


def score_response_variance(df: pd.DataFrame) -> float:
    """
    Reliability score from repeated outputs for variance prompts.
    Higher similarity = lower variance = better reliability.
    """
    scores = []
    for _, grp in df.groupby(["prompt_index"]):
        outputs = grp["output"].dropna().tolist()
        if len(outputs) >= 2:
            scores.append(pairwise_mean_similarity(outputs))
    return safe_mean(scores)


def score_parameter_sensitivity(df: pd.DataFrame) -> float:
    """
    Similarity across temperature settings for the same prompt.
    """
    scores = []
    for _, grp in df.groupby(["prompt_index"]):
        outputs = grp["output"].dropna().tolist()
        if len(outputs) >= 2:
            scores.append(pairwise_mean_similarity(outputs))
    return safe_mean(scores)


# ============================================================
# Main
# ============================================================
def main():
    if not RAW_OUTPUT_FILE.exists():
        raise FileNotFoundError(f"Missing raw outputs file: {RAW_OUTPUT_FILE}")

    df = pd.read_csv(RAW_OUTPUT_FILE)
    df["output"] = df["output"].fillna("").astype(str)

    category_scores = {}

    def subset(cat):
        return df[df["category"] == cat].copy()

    # Each score is on 0–1 scale internally
    category_scores["consistency"] = score_consistency(subset("consistency"))
    category_scores["robustness"] = score_robustness(subset("robustness"))
    category_scores["reasoning_stability"] = score_reasoning_stability(subset("reasoning_stability"))
    category_scores["long_context"] = score_long_context(subset("long_context"))
    category_scores["edge_case"] = score_edge_case(subset("edge_case"))
    category_scores["response_variance"] = score_response_variance(subset("response_variance"))
    category_scores["parameter_sensitivity"] = score_parameter_sensitivity(subset("parameter_sensitivity"))

    # Weighted final score on 0–100 scale
    final_score = 0.0
    detailed_rows = []

    for category, weight in WEIGHTS.items():
        score_01 = category_scores.get(category, 0.0)
        score_100 = pct(score_01)
        contribution = score_100 * weight
        final_score += contribution

        detailed_rows.append(
            {
                "category": category,
                "score_0_to_1": round(score_01, 4),
                "score_0_to_100": score_100,
                "weight": weight,
                "weighted_contribution": round(contribution, 4),
            }
        )

    detailed_rows.append(
        {
            "category": "final_reliability_score",
            "score_0_to_1": round(final_score / 100.0, 4),
            "score_0_to_100": round(final_score, 2),
            "weight": 1.0,
            "weighted_contribution": round(final_score, 4),
        }
    )

    out_df = pd.DataFrame(detailed_rows)
    out_df.to_csv(SCORES_FILE, index=False)

    summary = {
        "category_scores_0_to_100": {k: pct(v) for k, v in category_scores.items()},
        "weights": WEIGHTS,
        "final_reliability_score_0_to_100": round(final_score, 2),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nCategory scores (0–100):")
    for k, v in summary["category_scores_0_to_100"].items():
        print(f"  {k}: {v}")

    print(f"\nFinal weighted reliability score: {summary['final_reliability_score_0_to_100']}/100")
    print(f"\nSaved detailed scores to: {SCORES_FILE}")
    print(f"Saved summary JSON to: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()