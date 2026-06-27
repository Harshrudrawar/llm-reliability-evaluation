import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

# ============================================================
# Config
# ============================================================
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RUNS = int(os.getenv("RUNS_PER_PROMPT", "5"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
TEMPERATURES = [0.0, 0.2, 0.7, 1.0]
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "2.0"))
SEED = int(os.getenv("SEED", "42"))

PROMPTS_FILE = Path("prompts.json")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RAW_OUTPUT_FILE = RESULTS_DIR / "raw_outputs.csv"
MANIFEST_FILE = RESULTS_DIR / "experiment_manifest.json"

client = OpenAI()
random.seed(SEED)

CSV_FIELDS = [
    "category",
    "prompt_index",
    "question_index",
    "run",
    "temperature",
    "phase",
    "prompt",
    "context",
    "question",
    "challenge",
    "reference",
    "previous_response_id",
    "response_id",
    "output",
]


# ============================================================
# Helpers
# ============================================================
def load_prompts(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stringify_reference(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def call_model(
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    previous_response_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Call the Responses API with lightweight retry handling."""
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {
                "model": MODEL,
                "input": prompt,
                "temperature": temperature,
            }
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id

            response = client.responses.create(**kwargs)
            output_text = (response.output_text or "").strip()
            response_id = getattr(response, "id", "")
            return output_text, response_id
        except Exception as exc:  # pragma: no cover - surfaced to user if API fails
            last_error = exc
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Model call failed after {MAX_RETRIES} attempts: {exc}") from exc
            sleep_for = RETRY_BACKOFF_SECONDS * attempt
            time.sleep(sleep_for)

    raise RuntimeError(f"Model call failed: {last_error}")


def append_row(rows: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
    rows.append(row)


def make_row(
    *,
    category: str,
    prompt_index: int,
    question_index: Any = "",
    run: Any = 1,
    temperature: Any = DEFAULT_TEMPERATURE,
    phase: str,
    prompt: str,
    context: str = "",
    question: str = "",
    challenge: str = "",
    reference: Any = "",
    previous_response_id: str = "",
    response_id: str = "",
    output: str = "",
) -> Dict[str, Any]:
    return {
        "category": category,
        "prompt_index": prompt_index,
        "question_index": question_index,
        "run": run,
        "temperature": temperature,
        "phase": phase,
        "prompt": prompt,
        "context": context,
        "question": question,
        "challenge": challenge,
        "reference": stringify_reference(reference),
        "previous_response_id": previous_response_id,
        "response_id": response_id,
        "output": output,
    }


def save_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_manifest(prompts_data: Dict[str, Any], path: Path) -> None:
    manifest = {
        "model": MODEL,
        "runs_per_prompt": RUNS,
        "default_temperature": DEFAULT_TEMPERATURE,
        "temperatures": TEMPERATURES,
        "seed": SEED,
        "categories": sorted(prompts_data.keys()),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def get_item_prompt(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("prompt", "")).strip()
    raise TypeError(f"Unsupported prompt item type: {type(item)!r}")


def get_dict_item(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return {"prompt": item}
    raise TypeError(f"Unsupported prompt item type: {type(item)!r}")


def make_long_context_prompt(context: str, question: str) -> str:
    return "Context:\n" + context + "\n\nQuestion:\n" + question


# ============================================================
# Experiment runners
# ============================================================
def run_consistency(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    for prompt_idx, item in enumerate(items, start=1):
        prompt = get_item_prompt(item)
        for run_idx in range(1, RUNS + 1):
            output, response_id = call_model(prompt, temperature=DEFAULT_TEMPERATURE)
            append_row(rows, make_row(
                category="consistency",
                prompt_index=prompt_idx,
                run=run_idx,
                phase="single_turn",
                prompt=prompt,
                output=output,
                response_id=response_id,
            ))


def run_robustness(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    for prompt_idx, item in enumerate(items, start=1):
        prompt = get_item_prompt(item)
        output, response_id = call_model(prompt, temperature=DEFAULT_TEMPERATURE)
        append_row(rows, make_row(
            category="robustness",
            prompt_index=prompt_idx,
            run=1,
            phase="single_turn",
            prompt=prompt,
            output=output,
            response_id=response_id,
        ))


def run_reasoning_stability(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    for item_idx, item in enumerate(items, start=1):
        spec = get_dict_item(item)
        prompt = str(spec.get("prompt", "")).strip()
        challenge = str(spec.get("challenge", "")).strip()
        answer = stringify_reference(spec.get("answer", ""))

        last_response_id = ""
        for run_idx in range(1, RUNS + 1):
            output, response_id = call_model(prompt, temperature=DEFAULT_TEMPERATURE)
            last_response_id = response_id
            append_row(rows, make_row(
                category="reasoning_stability",
                prompt_index=item_idx,
                run=run_idx,
                phase="initial",
                prompt=prompt,
                challenge=challenge,
                reference=answer,
                output=output,
                response_id=response_id,
            ))

        challenged_prompt = challenge or "Please reconsider your answer."
        challenged_output, challenged_response_id = call_model(
            challenged_prompt,
            temperature=DEFAULT_TEMPERATURE,
            previous_response_id=last_response_id,
        )
        append_row(rows, make_row(
            category="reasoning_stability",
            prompt_index=item_idx,
            run=1,
            phase="challenge",
            prompt=prompt,
            challenge=challenged_prompt,
            reference=answer,
            previous_response_id=last_response_id,
            output=challenged_output,
            response_id=challenged_response_id,
        ))


def run_long_context(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    for item_idx, item in enumerate(items, start=1):
        spec = get_dict_item(item)
        context = str(spec.get("context", "")).strip()
        questions = spec.get("questions", [])

        for q_idx, qspec in enumerate(questions, start=1):
            qspec = get_dict_item(qspec)
            question = str(qspec.get("question", "")).strip()
            answers = qspec.get("answers", [])
            full_prompt = make_long_context_prompt(context, question)
            output, response_id = call_model(full_prompt, temperature=DEFAULT_TEMPERATURE)
            append_row(rows, make_row(
                category="long_context",
                prompt_index=item_idx,
                question_index=q_idx,
                run=1,
                phase="long_context",
                prompt=full_prompt,
                context=context,
                question=question,
                reference=answers,
                output=output,
                response_id=response_id,
            ))


def run_edge_case(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    for prompt_idx, item in enumerate(items, start=1):
        spec = get_dict_item(item)
        prompt = str(spec.get("prompt", "")).strip()
        expected = stringify_reference(spec.get("expected", ""))
        output, response_id = call_model(prompt, temperature=DEFAULT_TEMPERATURE)
        append_row(rows, make_row(
            category="edge_case",
            prompt_index=prompt_idx,
            run=1,
            phase="single_turn",
            prompt=prompt,
            reference=expected,
            output=output,
            response_id=response_id,
        ))


def run_uncertainty_calibration(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    for prompt_idx, item in enumerate(items, start=1):
        spec = get_dict_item(item)
        prompt = str(spec.get("prompt", "")).strip()
        expected = stringify_reference(spec.get("expected", ""))
        output, response_id = call_model(prompt, temperature=DEFAULT_TEMPERATURE)
        append_row(rows, make_row(
            category="uncertainty_calibration",
            prompt_index=prompt_idx,
            run=1,
            phase="single_turn",
            prompt=prompt,
            reference=expected,
            output=output,
            response_id=response_id,
        ))


def run_user_pressure(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    for item_idx, item in enumerate(items, start=1):
        spec = get_dict_item(item)
        prompt = str(spec.get("prompt", "")).strip()
        challenge = str(spec.get("challenge", "")).strip()
        answer = stringify_reference(spec.get("answer", ""))

        last_response_id = ""
        for run_idx in range(1, RUNS + 1):
            output, response_id = call_model(prompt, temperature=DEFAULT_TEMPERATURE)
            last_response_id = response_id
            append_row(rows, make_row(
                category="user_pressure",
                prompt_index=item_idx,
                run=run_idx,
                phase="initial",
                prompt=prompt,
                challenge=challenge,
                reference=answer,
                output=output,
                response_id=response_id,
            ))

        pressure_prompt = challenge or "Please confirm my incorrect statement."
        pressured_output, pressured_response_id = call_model(
            pressure_prompt,
            temperature=DEFAULT_TEMPERATURE,
            previous_response_id=last_response_id,
        )
        append_row(rows, make_row(
            category="user_pressure",
            prompt_index=item_idx,
            run=1,
            phase="challenge",
            prompt=prompt,
            challenge=pressure_prompt,
            reference=answer,
            previous_response_id=last_response_id,
            output=pressured_output,
            response_id=pressured_response_id,
        ))


def run_response_variance(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    for prompt_idx, item in enumerate(items, start=1):
        prompt = get_item_prompt(item)
        for run_idx in range(1, RUNS + 1):
            output, response_id = call_model(prompt, temperature=DEFAULT_TEMPERATURE)
            append_row(rows, make_row(
                category="response_variance",
                prompt_index=prompt_idx,
                run=run_idx,
                phase="variance",
                prompt=prompt,
                output=output,
                response_id=response_id,
            ))


def run_parameter_sensitivity(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    for prompt_idx, item in enumerate(items, start=1):
        prompt = get_item_prompt(item)
        for temp in TEMPERATURES:
            output, response_id = call_model(prompt, temperature=temp)
            append_row(rows, make_row(
                category="parameter_sensitivity",
                prompt_index=prompt_idx,
                run=1,
                temperature=temp,
                phase="temperature_sweep",
                prompt=prompt,
                output=output,
                response_id=response_id,
            ))


# ============================================================
# Main
# ============================================================
def main() -> None:
    prompts_data = load_prompts(PROMPTS_FILE)
    rows: List[Dict[str, Any]] = []

    if "consistency" in prompts_data:
        run_consistency(prompts_data["consistency"], rows)
    if "robustness" in prompts_data:
        run_robustness(prompts_data["robustness"], rows)
    if "reasoning_stability" in prompts_data:
        run_reasoning_stability(prompts_data["reasoning_stability"], rows)
    if "long_context" in prompts_data:
        run_long_context(prompts_data["long_context"], rows)
    if "edge_case" in prompts_data:
        run_edge_case(prompts_data["edge_case"], rows)
    if "uncertainty_calibration" in prompts_data:
        run_uncertainty_calibration(prompts_data["uncertainty_calibration"], rows)
    if "user_pressure" in prompts_data:
        run_user_pressure(prompts_data["user_pressure"], rows)
    if "response_variance" in prompts_data:
        run_response_variance(prompts_data["response_variance"], rows)
    if "parameter_sensitivity" in prompts_data:
        run_parameter_sensitivity(prompts_data["parameter_sensitivity"], rows)

    save_rows(rows, RAW_OUTPUT_FILE)
    save_manifest(prompts_data, MANIFEST_FILE)
    print(f"Saved {len(rows)} rows to {RAW_OUTPUT_FILE}")
    print(f"Saved experiment manifest to {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
