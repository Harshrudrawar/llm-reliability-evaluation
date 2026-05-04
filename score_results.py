import json
import os
import re
from collections import Counter
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

# ============================================================
# Config
# ============================================================
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

RESULTS_DIR = Path("results")
RAW_OUTPUT_FILE = RESULTS_DIR / "raw_outputs.csv"
SCORES_FILE = RESULTS_DIR / "scores.csv"
SUMMARY_JSON = RESULTS_DIR / "summary.json"

client = OpenAI()

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
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@lru_cache(maxsize=8192)
def get_embedding(text: str) -> Tuple[float, ...]:
    """
    Cached embedding lookup for one normalized text string.
    """
    text = normalize_text(text)
    if not text:
        return tuple()

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        encoding_format="float",
    )
    return tuple(response.data[0].embedding)


def cosine_similarity_text(a: str, b: str) -> float:
    """
    Cosine similarity between two texts using OpenAI embeddings.
    OpenAI embeddings are normalized, so cosine / dot-product ranking is appropriate.
    """
    va = np.array(get_embedding(a), dtype=np.float32)
    vb = np.array(get_embedding(b), dtype=np.float32)

    if va.size == 0 or vb.size == 0:
        return 0.0

    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def pairwise_mean_cosine(texts: List[str]) -> float:
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) == 0:
        return 0.0
    if len(texts) == 1:
        return 1.0

    scores = []
    for a, b in combinations(texts, 2):
        scores.append(cosine_similarity_text(a, b))

    return float(np.mean(scores)) if scores else 0.0


def safe_mean(values: List[float]) -> float:
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else 0.0


def pct(x: float) -> float:
    return round(x * 100.0, 2)


# ============================================================
# Task-specific helpers
# ============================================================
def classify_reasoning_answer(text: str) -> str:
    """
    Coarse labels for the diamond puzzle.
    """
    t = normalize_text(text)

    if "bed" in t and "cup" in t:
        return "ambiguous"
    if "bed" in t:
        return "bed"
    if "cup" in t:
        return "cup"
    if "fridge" in t:
        return "fridge"
    return "other"


def matches_long_context_reference(question: str, output: str) -> float:
    """
    Hybrid score: cosine similarity + keyword check for the long-context task.
    Returns a score in [0,1].
    """
    q = normalize_text(question)
    o = normalize_text(output)

    if "best title" in q or "main idea" in q:
        refs = [
            "Great Art: Business as Usual",
            "Great art is business as usual",
            "Art has become business as usual",
        ]
        keyword_hits = [
            ("great art" in o and "business" in o),
            ("business as usual" in o),
            ("art as business" in o),
        ]
    elif "where was" in q and "exhibited" in q:
        refs = [
            "The Metropolitan Museum of Modern Art",
            "Metropolitan Museum of Modern Art",
            "Met Museum",
        ]
        keyword_hits = [
            ("metropolitan museum of modern art" in o),
            ("met museum" in o),
            ("metropolitan museum" in o),
        ]
    elif "main argument" in q:
        refs = [
            "Art is becoming a commercial commodity and big business",
            "The passage argues that great art is turning into big business",
            "Art is treated like a commodity",
        ]
        keyword_hits = [
            ("commercial" in o),
            ("big business" in o),
            ("commodity" in o),
        ]
    else:
        refs = [q]
        keyword_hits = [False]

    cosine_best = max(cosine_similarity_text(o, ref) for ref in refs)
    keyword_best = 1.0 if any(keyword_hits) else 0.0

    # A pragmatic hybrid: keyword evidence OR high semantic match.
    if keyword_best > 0.0:
        return 1.0
    return 1.0 if cosine_best >= 0.74 else cosine_best


def edge_case_uncertainty_score(output: str) -> float:
    """
    Score future/unknown answers by checking uncertainty and semantic closeness.
    """
    o = normalize_text(output)
    uncertainty_markers = [
        "not been announced",
        "not yet announced",
        "has not been announced",
        "has not been awarded yet",
        "not yet occurred",
        "has not happened yet",
        "no official",
        "cannot know",
        "not available",
    ]

    if any(marker in o for marker in uncertainty_markers):
        return 1.0

    ref = "The Nobel Prize winners for 2030 have not been announced yet."
    sim = cosine_similarity_text(o, ref)
    return 1.0 if sim >= 0.72 else sim


# ============================================================
# Category scorers
# ============================================================
def score_consistency(df: pd.DataFrame) -> float:
    """
    Average pairwise cosine similarity across repeated outputs for the same prompt.
    """
    scores = []
    for _, grp in df.groupby("prompt_index"):
        outputs = grp["output"].dropna().astype(str).tolist()
        if len(outputs) >= 1:
            scores.append(pairwise_mean_cosine(outputs))
    return safe_mean(scores)


def score_robustness(df: pd.DataFrame) -> float:
    """
    Pairwise cosine similarity across paraphrased prompts.
    """
    outputs = df["output"].dropna().astype(str).tolist()
    return pairwise_mean_cosine(outputs)


def score_reasoning_stability(df: pd.DataFrame) -> float:
    """
    Hybrid score:
    - semantic similarity across responses
    - majority-label stability for the diamond puzzle
    """
    scores = []
    for _, grp in df.groupby("prompt_index"):
        outputs = grp["output"].dropna().astype(str).tolist()
        if not outputs:
            continue

        semantic_stability = pairwise_mean_cosine(outputs)

        labels = [classify_reasoning_answer(o) for o in outputs]
        labels = [l for l in labels if l != "other"]

        if labels:
            label_counts = Counter(labels)
            majority_fraction = label_counts.most_common(1)[0][1] / len(labels)
        else:
            majority_fraction = 0.0

        scores.append(0.5 * semantic_stability + 0.5 * majority_fraction)

    return safe_mean(scores)


def score_long_context(df: pd.DataFrame) -> float:
    """
    Score each long-context question against a reference answer.
    """
    scores = []
    for _, row in df.iterrows():
        question = str(row.get("question", ""))
        output = str(row.get("output", ""))
        if not question.strip():
            continue
        scores.append(matches_long_context_reference(question, output))
    return safe_mean(scores)


def score_edge_case(df: pd.DataFrame) -> float:
    """
    Score whether the model correctly handles the future/unknown question.
    """
    outputs = df["output"].dropna().astype(str).tolist()
    if not outputs:
        return 0.0
    return safe_mean([edge_case_uncertainty_score(o) for o in outputs])


def score_response_variance(df: pd.DataFrame) -> float:
    """
    Higher cosine similarity across repeated outputs = lower variance = better reliability.
    """
    scores = []
    for _, grp in df.groupby("prompt_index"):
        outputs = grp["output"].dropna().astype(str).tolist()
        if len(outputs) >= 1:
            scores.append(pairwise_mean_cosine(outputs))
    return safe_mean(scores)


def score_parameter_sensitivity(df: pd.DataFrame) -> float:
    """
    Similarity across temperature settings for the same prompt.
    """
    scores = []
    for _, grp in df.groupby("prompt_index"):
        outputs = grp["output"].dropna().astype(str).tolist()
        if len(outputs) >= 1:
            scores.append(pairwise_mean_cosine(outputs))
    return safe_mean(scores)


# ============================================================
# Main
# ============================================================
def main():
    if not RAW_OUTPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {RAW_OUTPUT_FILE}")

    df = pd.read_csv(RAW_OUTPUT_FILE)
    df["output"] = df["output"].fillna("").astype(str)

    def subset(cat: str) -> pd.DataFrame:
        return df[df["category"] == cat].copy()

    category_scores = {
        "consistency": score_consistency(subset("consistency")),
        "robustness": score_robustness(subset("robustness")),
        "reasoning_stability": score_reasoning_stability(subset("reasoning_stability")),
        "long_context": score_long_context(subset("long_context")),
        "edge_case": score_edge_case(subset("edge_case")),
        "response_variance": score_response_variance(subset("response_variance")),
        "parameter_sensitivity": score_parameter_sensitivity(subset("parameter_sensitivity")),
    }

    # Weighted final score in [0, 100]
    weighted_score = 0.0
    detail_rows = []

    for category, weight in WEIGHTS.items():
        score_01 = category_scores[category]
        score_100 = pct(score_01)
        contribution = score_100 * weight
        weighted_score += contribution

        detail_rows.append(
            {
                "category": category,
                "score_0_to_1": round(score_01, 4),
                "score_0_to_100": score_100,
                "weight": weight,
                "weighted_contribution": round(contribution, 4),
            }
        )

    detail_rows.append(
        {
            "category": "final_reliability_score",
            "score_0_to_1": round(weighted_score / 100.0, 4),
            "score_0_to_100": round(weighted_score, 2),
            "weight": 1.0,
            "weighted_contribution": round(weighted_score, 4),
        }
    )

    out_df = pd.DataFrame(detail_rows)
    out_df.to_csv(SCORES_FILE, index=False)

    summary = {
        "category_scores_0_to_100": {k: pct(v) for k, v in category_scores.items()},
        "weights": WEIGHTS,
        "final_reliability_score_0_to_100": round(weighted_score, 2),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nCategory scores (0–100):")
    for k, v in summary["category_scores_0_to_100"].items():
        print(f"  {k}: {v}")

    print(f"\nFinal weighted reliability score: {summary['final_reliability_score_0_to_100']}/100")
    print(f"Saved detailed scores to: {SCORES_FILE}")
    print(f"Saved summary JSON to: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()