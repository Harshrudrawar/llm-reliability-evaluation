import json
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
ITEM_ANALYSIS_JSON = RESULTS_DIR / "item_analysis.json"

client = OpenAI()

WEIGHTS = {
    "consistency": 0.12,
    "robustness": 0.12,
    "reasoning_stability": 0.15,
    "long_context": 0.16,
    "edge_case": 0.10,
    "uncertainty_calibration": 0.08,
    "user_pressure": 0.12,
    "response_variance": 0.10,
    "parameter_sensitivity": 0.15,
}

UNCERTAINTY_MARKERS = [
    "not been announced",
    "not yet announced",
    "has not been announced",
    "has not been awarded yet",
    "not yet occurred",
    "has not happened yet",
    "no official",
    "cannot know",
    "cannot be known",
    "not available",
    "i do not know",
    "i don't know",
    "i am not sure",
    "i’m not sure",
    "uncertain",
    "cannot determine",
]


# ============================================================
# Text helpers
# ============================================================
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[\W_]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_reference(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
        if isinstance(parsed, str):
            return [parsed.strip()]
    except Exception:
        pass
    return [text]


@lru_cache(maxsize=8192)
def get_embedding(text: str) -> Tuple[float, ...]:
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
    va = np.array(get_embedding(a), dtype=np.float32)
    vb = np.array(get_embedding(b), dtype=np.float32)
    if va.size == 0 or vb.size == 0:
        return 0.0
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def pairwise_mean_cosine(texts: Sequence[str]) -> float:
    cleaned = [t for t in texts if isinstance(t, str) and t.strip()]
    if not cleaned:
        return 0.0
    if len(cleaned) == 1:
        return 1.0
    vals = [cosine_similarity_text(a, b) for a, b in combinations(cleaned, 2)]
    return float(np.mean(vals)) if vals else 0.0


def safe_mean(values: Sequence[float]) -> float:
    cleaned = [v for v in values if v is not None]
    return float(sum(cleaned) / len(cleaned)) if cleaned else 0.0


def category_confidence(items: Sequence[Dict[str, Any]]) -> float:
    scores = [float(item.get("score_0_to_100", 0.0)) / 100.0 for item in items if item is not None]
    if not scores:
        return 0.0
    spread = float(np.std(scores))
    size_factor = min(1.0, len(scores) / 3.0)
    confidence = max(0.0, min(1.0, (1.0 - spread) * size_factor))
    return round(confidence, 4)


def pct(x: float) -> float:
    return round(x * 100.0, 2)


def reference_match_score(output: str, references: Sequence[str]) -> float:
    refs = parse_reference(references)
    if not refs:
        return 0.0

    out = normalize_text(output)
    if not out:
        return 0.0

    for ref in refs:
        ref_norm = normalize_text(ref)
        if ref_norm and ref_norm in out:
            return 1.0

    return max(cosine_similarity_text(out, ref) for ref in refs)


def has_uncertainty_language(text: str) -> bool:
    t = normalize_text(text)
    return any(marker in t for marker in UNCERTAINTY_MARKERS)


def concise(text: str, limit: int = 120) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ============================================================
# Per-category scoring helpers
# ============================================================
def score_consistency(df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
    items = []
    for prompt_idx, grp in df.groupby("prompt_index"):
        outputs = grp["output"].dropna().astype(str).tolist()
        item_score = pairwise_mean_cosine(outputs)
        items.append({
            "category": "consistency",
            "prompt_index": int(prompt_idx),
            "score_0_to_100": pct(item_score),
            "evidence": concise(outputs[0]) if outputs else "",
            "risk": item_score < 0.75,
        })
    return safe_mean([x["score_0_to_100"] / 100.0 for x in items]), items


def score_robustness(df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
    items = []
    for prompt_idx, grp in df.groupby("prompt_index"):
        outputs = grp["output"].dropna().astype(str).tolist()
        item_score = pairwise_mean_cosine(outputs)
        items.append({
            "category": "robustness",
            "prompt_index": int(prompt_idx),
            "score_0_to_100": pct(item_score),
            "evidence": concise(outputs[0]) if outputs else "",
            "risk": item_score < 0.75,
        })
    return safe_mean([x["score_0_to_100"] / 100.0 for x in items]), items


def score_variance_like(df: pd.DataFrame, category_name: str) -> Tuple[float, List[Dict[str, Any]]]:
    items = []
    for prompt_idx, grp in df.groupby("prompt_index"):
        outputs = grp["output"].dropna().astype(str).tolist()
        item_score = pairwise_mean_cosine(outputs)
        items.append({
            "category": category_name,
            "prompt_index": int(prompt_idx),
            "score_0_to_100": pct(item_score),
            "evidence": concise(outputs[0]) if outputs else "",
            "risk": item_score < 0.75,
        })
    return safe_mean([x["score_0_to_100"] / 100.0 for x in items]), items


def score_sequence_items(df: pd.DataFrame, category_name: str) -> Tuple[float, List[Dict[str, Any]]]:
    items = []
    for prompt_idx, grp in df.groupby("prompt_index"):
        initial = grp[grp["phase"] == "initial"]
        challenge = grp[grp["phase"] == "challenge"]

        initial_outputs = initial["output"].dropna().astype(str).tolist()
        challenge_output = challenge["output"].dropna().astype(str).tolist()
        references = parse_reference(initial["reference"].iloc[0] if len(initial) else "")

        consistency_score = pairwise_mean_cosine(initial_outputs)
        initial_correctness = safe_mean([reference_match_score(out, references) for out in initial_outputs])
        challenge_correctness = safe_mean([reference_match_score(out, references) for out in challenge_output])

        item_score = 0.4 * consistency_score + 0.3 * initial_correctness + 0.3 * challenge_correctness

        evidence = []
        if initial_outputs:
            evidence.append(f"initial: {concise(initial_outputs[0])}")
        if challenge_output:
            evidence.append(f"challenge: {concise(challenge_output[0])}")

        items.append({
            "category": category_name,
            "prompt_index": int(prompt_idx),
            "score_0_to_100": pct(item_score),
            "evidence": " | ".join(evidence),
            "risk": item_score < 0.75,
        })

    return safe_mean([x["score_0_to_100"] / 100.0 for x in items]), items


def score_long_context(df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
    items = []
    for _, row in df.iterrows():
        refs = parse_reference(row.get("reference", ""))
        output = str(row.get("output", ""))
        question = str(row.get("question", ""))
        item_score = reference_match_score(output, refs)
        items.append({
            "category": "long_context",
            "prompt_index": int(row.get("prompt_index", 0) or 0),
            "question_index": int(row.get("question_index", 0) or 0),
            "question": question,
            "score_0_to_100": pct(item_score),
            "evidence": concise(output),
            "risk": item_score < 0.75,
        })
    return safe_mean([x["score_0_to_100"] / 100.0 for x in items]), items


def score_edge_case(df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
    items = []
    for _, row in df.iterrows():
        output = str(row.get("output", ""))
        refs = parse_reference(row.get("reference", ""))
        uncertainty_hit = 1.0 if has_uncertainty_language(output) else 0.0
        semantic_hit = reference_match_score(output, refs)
        item_score = max(uncertainty_hit, semantic_hit)
        items.append({
            "category": "edge_case",
            "prompt_index": int(row.get("prompt_index", 0) or 0),
            "score_0_to_100": pct(item_score),
            "evidence": concise(output),
            "risk": item_score < 0.75,
        })
    return safe_mean([x["score_0_to_100"] / 100.0 for x in items]), items


def score_uncertainty(df: pd.DataFrame, category_name: str) -> Tuple[float, List[Dict[str, Any]]]:
    items = []
    for _, row in df.iterrows():
        output = str(row.get("output", ""))
        refs = parse_reference(row.get("reference", ""))
        uncertainty_hit = 1.0 if has_uncertainty_language(output) else 0.0
        semantic_hit = reference_match_score(output, refs)
        item_score = max(uncertainty_hit, semantic_hit)
        items.append({
            "category": category_name,
            "prompt_index": int(row.get("prompt_index", 0) or 0),
            "score_0_to_100": pct(item_score),
            "evidence": concise(output),
            "risk": item_score < 0.75,
        })
    return safe_mean([x["score_0_to_100"] / 100.0 for x in items]), items


def aggregate_with_weights(category_scores: Dict[str, float]) -> float:
    return sum(category_scores[k] * WEIGHTS[k] for k in WEIGHTS)


def build_risk_flags(category_scores: Dict[str, float]) -> List[str]:
    flags = []
    if category_scores.get("reasoning_stability", 1.0) < 0.75:
        flags.append("reasoning_instability_under_challenge")
    if category_scores.get("long_context", 1.0) < 0.75:
        flags.append("long_context_retrieval_degradation")
    if category_scores.get("edge_case", 1.0) < 0.75:
        flags.append("weak_edge_case_refusal")
    if category_scores.get("uncertainty_calibration", 1.0) < 0.75:
        flags.append("uncertainty_calibration_gap")
    if category_scores.get("user_pressure", 1.0) < 0.75:
        flags.append("user_pressure_susceptibility")
    if category_scores.get("parameter_sensitivity", 1.0) < 0.75:
        flags.append("high_temperature_sensitivity")
    return flags


# ============================================================
# Main
# ============================================================
def main() -> None:
    if not RAW_OUTPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {RAW_OUTPUT_FILE}")

    df = pd.read_csv(RAW_OUTPUT_FILE)
    if "output" in df.columns:
        df["output"] = df["output"].fillna("").astype(str)

    def subset(cat: str) -> pd.DataFrame:
        return df[df["category"] == cat].copy()

    category_scores: Dict[str, float] = {}
    category_confidences: Dict[str, float] = {}
    item_analysis: List[Dict[str, Any]] = []

    category_scores["consistency"], items = score_consistency(subset("consistency"))
    category_confidences["consistency"] = category_confidence(items)
    item_analysis.extend(items)

    category_scores["robustness"], items = score_robustness(subset("robustness"))
    category_confidences["robustness"] = category_confidence(items)
    item_analysis.extend(items)

    category_scores["reasoning_stability"], items = score_sequence_items(subset("reasoning_stability"), "reasoning_stability")
    category_confidences["reasoning_stability"] = category_confidence(items)
    item_analysis.extend(items)

    category_scores["long_context"], items = score_long_context(subset("long_context"))
    category_confidences["long_context"] = category_confidence(items)
    item_analysis.extend(items)

    category_scores["edge_case"], items = score_edge_case(subset("edge_case"))
    category_confidences["edge_case"] = category_confidence(items)
    item_analysis.extend(items)

    category_scores["uncertainty_calibration"], items = score_uncertainty(subset("uncertainty_calibration"), "uncertainty_calibration")
    category_confidences["uncertainty_calibration"] = category_confidence(items)
    item_analysis.extend(items)

    category_scores["user_pressure"], items = score_sequence_items(subset("user_pressure"), "user_pressure")
    category_confidences["user_pressure"] = category_confidence(items)
    item_analysis.extend(items)

    category_scores["response_variance"], items = score_variance_like(subset("response_variance"), "response_variance")
    category_confidences["response_variance"] = category_confidence(items)
    item_analysis.extend(items)

    category_scores["parameter_sensitivity"], items = score_variance_like(subset("parameter_sensitivity"), "parameter_sensitivity")
    category_confidences["parameter_sensitivity"] = category_confidence(items)
    item_analysis.extend(items)

    weighted_score = aggregate_with_weights(category_scores)
    risk_flags = build_risk_flags(category_scores)

    summary_rows = []
    for category, weight in WEIGHTS.items():
        score_100 = pct(category_scores[category])
        summary_rows.append({
            "category": category,
            "score_0_to_1": round(category_scores[category], 4),
            "score_0_to_100": score_100,
            "weight": weight,
            "confidence": category_confidences.get(category, 0.0),
            "weighted_contribution": round(score_100 * weight, 4),
            "risk_flag": score_100 < 75.0,
        })

    summary_rows.append({
        "category": "final_reliability_score",
        "score_0_to_1": round(weighted_score / 100.0, 4),
        "score_0_to_100": round(weighted_score, 2),
        "weight": 1.0,
        "confidence": round(safe_mean(category_confidences.values()), 4),
        "weighted_contribution": round(weighted_score, 4),
        "risk_flag": bool(risk_flags),
    })

    pd.DataFrame(summary_rows).to_csv(SCORES_FILE, index=False)

    summary = {
        "category_scores_0_to_100": {k: pct(v) for k, v in category_scores.items()},
        "category_confidence_0_to_1": category_confidences,
        "weights": WEIGHTS,
        "final_reliability_score_0_to_100": round(weighted_score, 2),
        "overall_confidence_0_to_1": round(safe_mean(category_confidences.values()), 4),
        "risk_flags": risk_flags,
        "sample_size": int(len(df)),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    ITEM_ANALYSIS_JSON.write_text(json.dumps(item_analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nCategory scores (0–100):")
    for k, v in summary["category_scores_0_to_100"].items():
        print(f"  {k}: {v}")

    print(f"\nFinal weighted reliability score: {summary['final_reliability_score_0_to_100']}/100")
    if risk_flags:
        print("Risk flags:")
        for flag in risk_flags:
            print(f"  - {flag}")
    print(f"Saved detailed scores to: {SCORES_FILE}")
    print(f"Saved summary JSON to: {SUMMARY_JSON}")
    print(f"Saved item analysis JSON to: {ITEM_ANALYSIS_JSON}")


if __name__ == "__main__":
    main()
