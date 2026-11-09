"""Generates the counterfactual benchmark: for every (task template, demographic
condition, subject model, repeat), calls the subject model with the templated
prompt and stores the raw response in data/benchmark.duckdb.

This is the offline step that produces the "unique dataset" -- these model
responses don't exist anywhere until we generate them.

Usage:
    uv run scripts/build_templates.py          # writes data/templates.json, conditions.json
    uv run scripts/generate_benchmark.py --pilot              # 2 templates, all conditions/models, 1 repeat
    uv run scripts/generate_benchmark.py --repeats 3           # full run
"""

import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from db_schema import connect, DB_PATH_DEFAULT  # noqa: E402

load_dotenv(override=True)  # .env must win over a stale/invalid key already in the shell env

DATA_DIR = Path(__file__).parent.parent / "data"

# Subject models under study. Verify these IDs are live before a full paid
# run -- model lineups change; these are this project's current best guess
# at a cheap/fast tier from each vendor, plus a local open-weight model.
SUBJECT_MODELS = {
    "claude": "anthropic/claude-haiku-5-5",
    "gpt": "openai/gpt-5-mini",
    "llama": "ollama/llama3.1",
}

# gpt-5-mini is a reasoning model: by default it spends part of max_tokens on hidden
# reasoning tokens before writing anything, which can silently truncate the visible
# response to empty on harder prompts. "minimal" keeps it comparable to the
# non-reasoning subject models (claude, llama) -- a quick casual answer, not an
# overthought one.
EXTRA_PARAMS = {
    # gpt-5-mini rejects any temperature != 1 while reasoning is active, even at
    # reasoning_effort="minimal" -- override the shared default for this model only.
    "gpt": {"reasoning_effort": "minimal", "temperature": 1},
}

MAX_RETRIES = 2
# Both subject models reject temperature != 1 in their current API versions
# (claude-haiku-5-5 outright; gpt-5-mini while reasoning is active) -- repeats
# still vary at the default temperature, just not boosted above it.
TEMPERATURE = 1
MAX_TOKENS = 500
MAX_WORKERS = 8


def load_grid():
    templates = json.loads((DATA_DIR / "templates.json").read_text())
    conditions = json.loads((DATA_DIR / "conditions.json").read_text())
    return templates, conditions


def build_prompt(template_text: str, phrase: str) -> str:
    return f"{template_text} {phrase}".strip() if phrase else template_text


def call_model(model_id: str, prompt: str, extra_params: dict | None = None) -> str:
    last_err = None
    kwargs = {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "timeout": 60}
    kwargs.update(extra_params or {})
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = litellm.completion(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 -- retry anything transient, surface the last error
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise last_err


def generate_one(template, condition, model_key, model_id, generation_idx):
    prompt = build_prompt(template["text_template"], condition["phrase"])
    text = call_model(model_id, prompt, EXTRA_PARAMS.get(model_key))
    return {
        "response_id": str(uuid.uuid4()),
        "template_id": template["template_id"],
        "condition_id": condition["condition_id"],
        "model": model_key,
        "generation_idx": generation_idx,
        "prompt_text": prompt,
        "response_text": text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Only first 2 templates, 1 repeat.")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--models", type=str, default="claude,gpt")
    parser.add_argument("--db", type=str, default=DB_PATH_DEFAULT)
    args = parser.parse_args()

    templates, conditions = load_grid()
    if args.pilot:
        templates = templates[:2]
        repeats = 1
    else:
        repeats = args.repeats

    model_keys = [m.strip() for m in args.models.split(",")]
    models = {k: SUBJECT_MODELS[k] for k in model_keys}

    con = connect(args.db)
    con.executemany(
        "INSERT OR REPLACE INTO templates VALUES (?, ?, ?)",
        [(t["template_id"], t["task_category"], t["text_template"]) for t in templates],
    )
    con.executemany(
        "INSERT OR REPLACE INTO conditions VALUES (?, ?, ?, ?)",
        [(c["condition_id"], c["axis"], c["label"], c["phrase"]) for c in conditions],
    )

    jobs = [
        (template, condition, model_key, model_id, i)
        for template in templates
        for condition in conditions
        for model_key, model_id in models.items()
        for i in range(repeats)
    ]
    print(f"Running {len(jobs)} generations across {len(models)} models...")

    done, failed = 0, 0
    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(generate_one, template, condition, model_key, model_id, i): (template, condition, model_key, i)
            for template, condition, model_key, model_id, i in jobs
        }
        for future in as_completed(futures):
            template, condition, model_key, i = futures[future]
            try:
                rows.append(future.result())
                done += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAILED template={template['template_id']} condition={condition['condition_id']} "
                      f"model={model_key} rep={i}: {type(e).__name__}: {e}")
            if (done + failed) % 50 == 0:
                print(f"  {done + failed}/{len(jobs)} ({failed} failed)")

    con.executemany(
        "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, now())",
        [(r["response_id"], r["template_id"], r["condition_id"], r["model"],
          r["generation_idx"], r["prompt_text"], r["response_text"]) for r in rows],
    )
    con.close()
    print(f"Done. {done} succeeded, {failed} failed. Wrote to {args.db}")


if __name__ == "__main__":
    main()
