import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ============================================================
# Configuration
# ============================================================
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RUNS = int(os.getenv("RUNS_PER_PROMPT", "5"))
TEMPERATURE_DEFAULT = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
TEMPERATURES = [0.2, 0.7, 1.0]

PROMPTS_FILE = "prompts.json"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RAW_OUTPUT_FILE = RESULTS_DIR / "raw_outputs.csv"

client = OpenAI()


# ============================================================
# Utility functions
# ============================================================
def load_prompts(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_model(
    prompt: str,
    temperature: float = TEMPERATURE_DEFAULT,
    previous_response_id: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """
    Returns:
        output_text: the model's text response
        response_id: the response id for chaining stateful follow-ups
    """
    kwargs = {
        "model": MODEL,
        "input": prompt,
        "temperature": temperature,
    }
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    response = client.responses.create(**kwargs)
    output_text = (response.output_text or "").strip()
    response_id = getattr(response, "id", None)
    return output_text, response_id


def append_row(rows: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
    rows.append(row)


def save_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_long_context_prompt(context: str, question: str) -> str:
    return f"""Context:
{context}

Question:
{question}"""


# ============================================================
# Experiment runners
# ============================================================
def run_consistency(prompts: List[str], rows: List[Dict[str, Any]]) -> None:
    for prompt_idx, prompt in enumerate(prompts, start=1):
        for run_idx in range(1, RUNS + 1):
            output, response_id = call_model(prompt, temperature=TEMPERATURE_DEFAULT)
            append_row(
                rows,
                {
                    "category": "consistency",
                    "prompt_index": prompt_idx,
                    "run": run_idx,
                    "temperature": TEMPERATURE_DEFAULT,
                    "prompt": prompt,
                    "context": "",
                    "question": "",
                    "challenge": "",
                    "phase": "single_turn",
                    "previous_response_id": "",
                    "response_id": response_id or "",
                    "output": output,
                },
            )


def run_robustness(prompts: List[str], rows: List[Dict[str, Any]]) -> None:
    for prompt_idx, prompt in enumerate(prompts, start=1):
        output, response_id = call_model(prompt, temperature=TEMPERATURE_DEFAULT)
        append_row(
            rows,
            {
                "category": "robustness",
                "prompt_index": prompt_idx,
                "run": 1,
                "temperature": TEMPERATURE_DEFAULT,
                "prompt": prompt,
                "context": "",
                "question": "",
                "challenge": "",
                "phase": "single_turn",
                "previous_response_id": "",
                "response_id": response_id or "",
                "output": output,
            },
        )


def run_reasoning_stability(items: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    for item_idx, item in enumerate(items, start=1):
        prompt = item["prompt"]
        challenge = item.get("challenge", "wrong")

        # Initial repeated runs
        last_response_id = None
        initial_outputs = []

        for run_idx in range(1, RUNS + 1):
            output, response_id = call_model(prompt, temperature=TEMPERATURE_DEFAULT)
            initial_outputs.append(output)
            last_response_id = response_id

            append_row(
                rows,
                {
                    "category": "reasoning_stability",
                    "prompt_index": item_idx,
                    "run": run_idx,
                    "temperature": TEMPERATURE_DEFAULT,
                    "prompt": prompt,
                    "context": "",
                    "question": "",
                    "challenge": "",
                    "phase": "initial",
                    "previous_response_id": "",
                    "response_id": response_id or "",
                    "output": output,
                },
            )

        # Challenged follow-up using previous_response_id (stateful continuation)
        challenge_prompt = f"The user says: {challenge}"
        challenged_output, challenged_response_id = call_model(
            challenge_prompt,
            temperature=TEMPERATURE_DEFAULT,
            previous_response_id=last_response_id,
        )

        append_row(
            rows,
            {
                "category": "reasoning_stability",
                "prompt_index": item_idx,
                "run": 1,
                "temperature": TEMPERATURE_DEFAULT,
                "prompt": prompt,
                "context": "",
                "question": "",
                "challenge": challenge,
                "phase": "challenged",
                "previous_response_id": last_response_id or "",
                "response_id": challenged_response_id or "",
                "output": challenged_output,
            },
        )


def run_long_context(items: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    for item_idx, item in enumerate(items, start=1):
        context = item["context"]
        questions = item["questions"]

        for q_idx, question in enumerate(questions, start=1):
            full_prompt = make_long_context_prompt(context, question)
            output, response_id = call_model(full_prompt, temperature=TEMPERATURE_DEFAULT)

            append_row(
                rows,
                {
                    "category": "long_context",
                    "prompt_index": item_idx,
                    "question_index": q_idx,
                    "run": 1,
                    "temperature": TEMPERATURE_DEFAULT,
                    "prompt": full_prompt,
                    "context": context,
                    "question": question,
                    "challenge": "",
                    "phase": "long_context",
                    "previous_response_id": "",
                    "response_id": response_id or "",
                    "output": output,
                },
            )


def run_edge_case(prompts: List[str], rows: List[Dict[str, Any]]) -> None:
    for prompt_idx, prompt in enumerate(prompts, start=1):
        output, response_id = call_model(prompt, temperature=TEMPERATURE_DEFAULT)
        append_row(
            rows,
            {
                "category": "edge_case",
                "prompt_index": prompt_idx,
                "run": 1,
                "temperature": TEMPERATURE_DEFAULT,
                "prompt": prompt,
                "context": "",
                "question": "",
                "challenge": "",
                "phase": "single_turn",
                "previous_response_id": "",
                "response_id": response_id or "",
                "output": output,
            },
        )


def run_response_variance(prompts: List[str], rows: List[Dict[str, Any]]) -> None:
    for prompt_idx, prompt in enumerate(prompts, start=1):
        for run_idx in range(1, RUNS + 1):
            output, response_id = call_model(prompt, temperature=TEMPERATURE_DEFAULT)
            append_row(
                rows,
                {
                    "category": "response_variance",
                    "prompt_index": prompt_idx,
                    "run": run_idx,
                    "temperature": TEMPERATURE_DEFAULT,
                    "prompt": prompt,
                    "context": "",
                    "question": "",
                    "challenge": "",
                    "phase": "variance",
                    "previous_response_id": "",
                    "response_id": response_id or "",
                    "output": output,
                },
            )


def run_parameter_sensitivity(prompts: List[str], rows: List[Dict[str, Any]]) -> None:
    for prompt_idx, prompt in enumerate(prompts, start=1):
        for temp in TEMPERATURES:
            output, response_id = call_model(prompt, temperature=temp)
            append_row(
                rows,
                {
                    "category": "parameter_sensitivity",
                    "prompt_index": prompt_idx,
                    "run": 1,
                    "temperature": temp,
                    "prompt": prompt,
                    "context": "",
                    "question": "",
                    "challenge": "",
                    "phase": "parameter_sensitivity",
                    "previous_response_id": "",
                    "response_id": response_id or "",
                    "output": output,
                },
            )


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

    if "response_variance" in prompts_data:
        run_response_variance(prompts_data["response_variance"], rows)

    if "parameter_sensitivity" in prompts_data:
        run_parameter_sensitivity(prompts_data["parameter_sensitivity"], rows)

    save_rows(rows, RAW_OUTPUT_FILE)
    print(f"Saved {len(rows)} rows to {RAW_OUTPUT_FILE}")


if __name__ == "__main__":
    main()