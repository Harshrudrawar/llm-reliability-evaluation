import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from providers import get_provider

# ============================================================
# Config
# ============================================================
RUNS = int(os.getenv("RUNS_PER_PROMPT", "5"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
TEMPERATURES = [0.0, 0.2, 0.7, 1.0]
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "2.0"))
SEED = int(os.getenv("SEED", "42"))

# Observability / robustness knobs
PROGRESS_EVERY = int(os.getenv("PROGRESS_EVERY", "25"))  # print elapsed every N calls
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10"))          # flush partial CSV every N calls
FAILED_PLACEHOLDER = "[API_CALL_FAILED]"

PROMPTS_FILE = Path(os.getenv("PROMPTS_FILE", "prompts.json"))
# When RUN_LABEL is set, each run gets its own results/<RUN_LABEL>/ folder so
# multiple model runs never overwrite each other. Unset -> legacy results/ path.
RUN_LABEL = os.getenv("RUN_LABEL", "").strip()
RESULTS_DIR = Path("results") / RUN_LABEL if RUN_LABEL else Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_OUTPUT_FILE = RESULTS_DIR / "raw_outputs.csv"
MANIFEST_FILE = RESULTS_DIR / "experiment_manifest.json"

provider = get_provider()
random.seed(SEED)

CSV_FIELDS = [
    "category",
    "prompt_index",
    "group_id",
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
    "error",  # additive, trailing column; empty on success (existing columns unchanged)
]

BAR = "=" * 36


# ============================================================
# Run statistics / progress
# ============================================================
class RunStats:
    def __init__(self) -> None:
        self.total = 0
        self.success = 0
        self.failed = 0
        self.start = time.monotonic()


STATS = RunStats()


def fmt_elapsed(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}m {s % 60}s"


def banner(title: str) -> None:
    print()
    print(BAR)
    print(title)
    print(BAR, flush=True)


def log(message: str) -> None:
    print(message, flush=True)


def record_call(rows: List[Dict[str, Any]], ok: bool) -> None:
    """Bookkeeping after every API call: counts, periodic partial save, periodic timing."""
    STATS.total += 1
    if ok:
        STATS.success += 1
    else:
        STATS.failed += 1

    if SAVE_EVERY > 0 and STATS.total % SAVE_EVERY == 0:
        save_rows(rows, RAW_OUTPUT_FILE)

    if PROGRESS_EVERY > 0 and STATS.total % PROGRESS_EVERY == 0:
        elapsed = time.monotonic() - STATS.start
        log(f"Completed {STATS.total} API calls | elapsed {fmt_elapsed(elapsed)}")


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


def as_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return {"prompt": item}
    raise TypeError(f"Unsupported prompt item type: {type(item)!r}")


def get_item_prompt(item: Any) -> str:
    return str(as_dict(item).get("prompt", "")).strip()


def get_group_id(category: str, item: Any, prompt_index: int, fallback: Optional[str] = None) -> str:
    spec = as_dict(item)
    group_id = str(spec.get("group_id", "")).strip()
    if group_id:
        return group_id
    if fallback:
        return fallback
    return f"{category}:{prompt_index}"


def call_model(
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    previous_response_id: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Generate one response via the selected provider, with lightweight retry.

    Never raises: on permanent failure (after MAX_RETRIES, including timeouts) it
    returns the FAILED_PLACEHOLDER output and a non-empty error string so the
    benchmark can record the failure and continue. Returns (output, response_id, error).
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            output, response_id = provider.generate(
                prompt,
                temperature=temperature,
                previous_response_id=previous_response_id,
            )
            return output, response_id, ""
        except Exception as exc:  # noqa: BLE001 - any provider/transport error (incl. timeouts)
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    error = f"{type(last_error).__name__}: {last_error}"
    print(f"  [error] API call failed after {MAX_RETRIES} attempts: {error}", file=sys.stderr, flush=True)
    return FAILED_PLACEHOLDER, "", error


def append_row(rows: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
    rows.append(row)


def make_row(
    *,
    category: str,
    prompt_index: int,
    group_id: str = "",
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
    error: str = "",
) -> Dict[str, Any]:
    return {
        "category": category,
        "prompt_index": prompt_index,
        "group_id": group_id,
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
        "error": error,
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
        "run_label": RUN_LABEL,
        "provider": provider.name,
        "model": provider.model,
        "runs_per_prompt": RUNS,
        "default_temperature": DEFAULT_TEMPERATURE,
        "temperatures": TEMPERATURES,
        "seed": SEED,
        "categories": sorted(prompts_data.keys()),
        "output_files": {
            "raw_outputs_csv": str(RAW_OUTPUT_FILE),
            "manifest_json": str(MANIFEST_FILE),
        },
        "row_schema": CSV_FIELDS,
        "grouping": {
            "robustness": "shared group_id across paraphrases",
            "response_variance": "shared group_id across prompt variants",
            "parameter_sensitivity": "grouped by prompt across temperatures",
            "reasoning_stability": "grouped by prompt with initial/challenge phases",
            "user_pressure": "grouped by prompt with initial/challenge phases",
        },
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def make_long_context_prompt(context: str, question: str) -> str:
    return "Context:\n" + context + "\n\nQuestion:\n" + question


# ============================================================
# Experiment runners
# ============================================================
def run_consistency(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    n = len(items)
    for prompt_idx, item in enumerate(items, start=1):
        prompt = get_item_prompt(item)
        group_id = get_group_id("consistency", item, prompt_idx)
        for run_idx in range(1, RUNS + 1):
            log(f"Consistency prompt {prompt_idx}/{n} | run {run_idx}/{RUNS}")
            output, response_id, error = call_model(prompt, temperature=DEFAULT_TEMPERATURE)
            append_row(
                rows,
                make_row(
                    category="consistency",
                    prompt_index=prompt_idx,
                    group_id=group_id,
                    run=run_idx,
                    phase="single_turn",
                    prompt=prompt,
                    output=output,
                    response_id=response_id,
                    error=error,
                ),
            )
            record_call(rows, error == "")


def run_robustness(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    """Run each paraphrase once, but give the scorer a shared group_id for the set."""
    n = len(items)
    for prompt_idx, item in enumerate(items, start=1):
        prompt = get_item_prompt(item)
        group_id = get_group_id("robustness", item, prompt_idx, fallback="robustness:shared")
        log(f"Robustness: {prompt_idx}/{n}")
        output, response_id, error = call_model(prompt, temperature=DEFAULT_TEMPERATURE)
        append_row(
            rows,
            make_row(
                category="robustness",
                prompt_index=prompt_idx,
                group_id=group_id,
                run=1,
                phase="single_turn",
                prompt=prompt,
                output=output,
                response_id=response_id,
                error=error,
            ),
        )
        record_call(rows, error == "")


def run_reasoning_stability(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    n = len(items)
    for item_idx, item in enumerate(items, start=1):
        spec = as_dict(item)
        prompt = str(spec.get("prompt", "")).strip()
        challenge = str(spec.get("challenge", "")).strip()
        answer = stringify_reference(spec.get("answer", ""))
        group_id = get_group_id("reasoning_stability", item, item_idx)

        last_response_id = ""
        for run_idx in range(1, RUNS + 1):
            log(f"Reasoning prompt {item_idx}/{n} | initial run {run_idx}/{RUNS}")
            output, response_id, error = call_model(prompt, temperature=DEFAULT_TEMPERATURE)
            last_response_id = response_id
            append_row(
                rows,
                make_row(
                    category="reasoning_stability",
                    prompt_index=item_idx,
                    group_id=group_id,
                    run=run_idx,
                    phase="initial",
                    prompt=prompt,
                    challenge=challenge,
                    reference=answer,
                    output=output,
                    response_id=response_id,
                    error=error,
                ),
            )
            record_call(rows, error == "")

        challenged_prompt = challenge or "Please reconsider your answer."
        log(f"Reasoning prompt {item_idx}/{n} | challenge")
        challenged_output, challenged_response_id, error = call_model(
            challenged_prompt,
            temperature=DEFAULT_TEMPERATURE,
            previous_response_id=last_response_id,
        )
        append_row(
            rows,
            make_row(
                category="reasoning_stability",
                prompt_index=item_idx,
                group_id=group_id,
                run=1,
                phase="challenge",
                prompt=prompt,
                challenge=challenged_prompt,
                reference=answer,
                previous_response_id=last_response_id,
                output=challenged_output,
                response_id=challenged_response_id,
                error=error,
            ),
        )
        record_call(rows, error == "")


def run_long_context(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    n = len(items)
    for item_idx, item in enumerate(items, start=1):
        spec = as_dict(item)
        context = str(spec.get("context", "")).strip()
        questions = spec.get("questions", [])
        group_id = get_group_id("long_context", item, item_idx)
        q_total = len(questions)

        for q_idx, qspec in enumerate(questions, start=1):
            qspec = as_dict(qspec)
            question = str(qspec.get("question", "")).strip()
            answers = qspec.get("answers", [])
            full_prompt = make_long_context_prompt(context, question)
            log(f"Long Context passage {item_idx}/{n} | question {q_idx}/{q_total}")
            output, response_id, error = call_model(full_prompt, temperature=DEFAULT_TEMPERATURE)
            append_row(
                rows,
                make_row(
                    category="long_context",
                    prompt_index=item_idx,
                    group_id=group_id,
                    question_index=q_idx,
                    run=1,
                    phase="long_context",
                    prompt=full_prompt,
                    context=context,
                    question=question,
                    reference=answers,
                    output=output,
                    response_id=response_id,
                    error=error,
                ),
            )
            record_call(rows, error == "")


def run_edge_case(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    n = len(items)
    for prompt_idx, item in enumerate(items, start=1):
        spec = as_dict(item)
        prompt = str(spec.get("prompt", "")).strip()
        expected = stringify_reference(spec.get("expected", ""))
        group_id = get_group_id("edge_case", item, prompt_idx)
        log(f"Edge Case: {prompt_idx}/{n}")
        output, response_id, error = call_model(prompt, temperature=DEFAULT_TEMPERATURE)
        append_row(
            rows,
            make_row(
                category="edge_case",
                prompt_index=prompt_idx,
                group_id=group_id,
                run=1,
                phase="single_turn",
                prompt=prompt,
                reference=expected,
                output=output,
                response_id=response_id,
                error=error,
            ),
        )
        record_call(rows, error == "")


def run_uncertainty_calibration(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    n = len(items)
    for prompt_idx, item in enumerate(items, start=1):
        spec = as_dict(item)
        prompt = str(spec.get("prompt", "")).strip()
        expected = stringify_reference(spec.get("expected", ""))
        group_id = get_group_id("uncertainty_calibration", item, prompt_idx)
        log(f"Uncertainty Calibration: {prompt_idx}/{n}")
        output, response_id, error = call_model(prompt, temperature=DEFAULT_TEMPERATURE)
        append_row(
            rows,
            make_row(
                category="uncertainty_calibration",
                prompt_index=prompt_idx,
                group_id=group_id,
                run=1,
                phase="single_turn",
                prompt=prompt,
                reference=expected,
                output=output,
                response_id=response_id,
                error=error,
            ),
        )
        record_call(rows, error == "")


def run_user_pressure(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    n = len(items)
    for item_idx, item in enumerate(items, start=1):
        spec = as_dict(item)
        prompt = str(spec.get("prompt", "")).strip()
        challenge = str(spec.get("challenge", "")).strip()
        answer = stringify_reference(spec.get("answer", ""))
        group_id = get_group_id("user_pressure", item, item_idx)

        last_response_id = ""
        for run_idx in range(1, RUNS + 1):
            log(f"User Pressure prompt {item_idx}/{n} | initial run {run_idx}/{RUNS}")
            output, response_id, error = call_model(prompt, temperature=DEFAULT_TEMPERATURE)
            last_response_id = response_id
            append_row(
                rows,
                make_row(
                    category="user_pressure",
                    prompt_index=item_idx,
                    group_id=group_id,
                    run=run_idx,
                    phase="initial",
                    prompt=prompt,
                    challenge=challenge,
                    reference=answer,
                    output=output,
                    response_id=response_id,
                    error=error,
                ),
            )
            record_call(rows, error == "")

        pressure_prompt = challenge or "Please confirm my incorrect statement."
        log(f"User Pressure prompt {item_idx}/{n} | challenge")
        pressured_output, pressured_response_id, error = call_model(
            pressure_prompt,
            temperature=DEFAULT_TEMPERATURE,
            previous_response_id=last_response_id,
        )
        append_row(
            rows,
            make_row(
                category="user_pressure",
                prompt_index=item_idx,
                group_id=group_id,
                run=1,
                phase="challenge",
                prompt=prompt,
                challenge=pressure_prompt,
                reference=answer,
                previous_response_id=last_response_id,
                output=pressured_output,
                response_id=pressured_response_id,
                error=error,
            ),
        )
        record_call(rows, error == "")


def run_response_variance(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    """Run variants under a shared group so the scorer can compare phrasing diversity."""
    n = len(items)
    shared_group = "response_variance:shared"
    for prompt_idx, item in enumerate(items, start=1):
        prompt = get_item_prompt(item)
        group_id = get_group_id("response_variance", item, prompt_idx, fallback=shared_group)
        for run_idx in range(1, RUNS + 1):
            log(f"Response Variance prompt {prompt_idx}/{n} | run {run_idx}/{RUNS}")
            output, response_id, error = call_model(prompt, temperature=DEFAULT_TEMPERATURE)
            append_row(
                rows,
                make_row(
                    category="response_variance",
                    prompt_index=prompt_idx,
                    group_id=group_id,
                    run=run_idx,
                    phase="variance",
                    prompt=prompt,
                    output=output,
                    response_id=response_id,
                    error=error,
                ),
            )
            record_call(rows, error == "")


def run_parameter_sensitivity(items: Sequence[Any], rows: List[Dict[str, Any]]) -> None:
    n = len(items)
    for prompt_idx, item in enumerate(items, start=1):
        prompt = get_item_prompt(item)
        group_id = get_group_id("parameter_sensitivity", item, prompt_idx)
        for temp in TEMPERATURES:
            log(f"Parameter Sensitivity prompt {prompt_idx}/{n} | temperature {temp}")
            output, response_id, error = call_model(prompt, temperature=temp)
            append_row(
                rows,
                make_row(
                    category="parameter_sensitivity",
                    prompt_index=prompt_idx,
                    group_id=group_id,
                    run=1,
                    temperature=temp,
                    phase="temperature_sweep",
                    prompt=prompt,
                    output=output,
                    response_id=response_id,
                    error=error,
                ),
            )
            record_call(rows, error == "")


# Ordered dispatch table: (json key, display name, runner). Order/logic preserved.
RUNNERS = [
    ("consistency", "Consistency", run_consistency),
    ("robustness", "Robustness", run_robustness),
    ("reasoning_stability", "Reasoning Stability", run_reasoning_stability),
    ("long_context", "Long Context", run_long_context),
    ("edge_case", "Edge Case", run_edge_case),
    ("uncertainty_calibration", "Uncertainty Calibration", run_uncertainty_calibration),
    ("user_pressure", "User Pressure", run_user_pressure),
    ("response_variance", "Response Variance", run_response_variance),
    ("parameter_sensitivity", "Parameter Sensitivity", run_parameter_sensitivity),
]


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("=" * 40)
    print("Run Label:")
    print(RUN_LABEL if RUN_LABEL else "(default - results/)")
    print("=" * 40, flush=True)

    prompts_data = load_prompts(PROMPTS_FILE)
    rows: List[Dict[str, Any]] = []
    STATS.start = time.monotonic()

    for key, display, runner in RUNNERS:
        if key in prompts_data:
            banner(f"Running {display}...")
            runner(prompts_data[key], rows)

    save_rows(rows, RAW_OUTPUT_FILE)
    save_manifest(prompts_data, MANIFEST_FILE)

    elapsed = time.monotonic() - STATS.start
    print()
    print(BAR)
    print("Benchmark Complete")
    print(BAR)
    print()
    print(f"Total API calls: {STATS.total}")
    print(f"Successful:      {STATS.success}")
    print(f"Failed:          {STATS.failed}")
    print(f"Elapsed time:    {fmt_elapsed(elapsed)}")
    print()
    print(f"Saved {len(rows)} rows to {RAW_OUTPUT_FILE}")
    print(f"Saved experiment manifest to {MANIFEST_FILE}")
    print()
    print("Results saved to:")
    print()
    print(f"{RESULTS_DIR.as_posix()}/")


if __name__ == "__main__":
    main()
