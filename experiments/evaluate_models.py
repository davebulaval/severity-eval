"""Pipeline d'évaluation des modèles via API avec scoring et logging wandb.

Usage:
    python -m experiments.evaluate_models --dataset financebench --model gpt-4o --limit 100
    python -m experiments.evaluate_models --dataset all --model all
    python -m experiments.evaluate_models --dataset financebench --model all --wandb
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models — OpenRouter replaces Together for Llama-3
# ---------------------------------------------------------------------------

MODELS = {
    "gpt-4o": {"provider": "openai", "model_id": "gpt-4o"},
    "claude-sonnet": {"provider": "anthropic", "model_id": "claude-sonnet-4-6"},
    "llama-3-70b": {"provider": "openrouter", "model_id": "meta-llama/llama-3-70b-instruct"},
    "mistral-large": {"provider": "mistral", "model_id": "mistral-large-latest"},
    "gemini-pro": {"provider": "google", "model_id": "gemini-1.5-pro"},
}

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

DATASETS = {
    # Finance
    "financebench": "experiments.datasets.load_financebench:load_financebench",
    "finqa": "experiments.datasets.load_finqa:load_finqa",
    "tatqa": "experiments.datasets.load_tatqa:load_tatqa",
    # Medical
    "medcalc": "experiments.datasets.load_medcalc:load_medcalc",
    "medqa": "experiments.datasets.load_medqa:load_medqa",
    "headqa": "experiments.datasets.load_headqa:load_headqa",
    # Legal
    "cuad": "experiments.datasets.load_cuad:load_cuad",
    "maud": "experiments.datasets.load_maud:load_maud",
    "contractnli": "experiments.datasets.load_contractnli:load_contractnli",
    # Insurance (local)
    "rag_insurance": "experiments.datasets.load_rag_insurance:load_rag_insurance",
    "judgebert": "experiments.datasets.load_judgebert:load_judgebert",
}

OUTPUT_DIR = Path("experiments/results")

# ---------------------------------------------------------------------------
# Prompts by domain
# ---------------------------------------------------------------------------

DOMAIN_PROMPTS = {
    "finance": (
        "Answer the following financial question concisely and accurately. "
        "Provide only the answer, no explanation.\n\nQuestion: {question}\nAnswer:"
    ),
    "medical": (
        "Answer the following medical question. "
        "Provide only the answer (numerical with units if applicable).\n\nQuestion: {question}\nAnswer:"
    ),
    "legal": (
        "Answer the following legal contract question. "
        "Provide only the relevant text or 'N/A' if not found.\n\nQuestion: {question}\nAnswer:"
    ),
    "insurance": (
        "Répondez à la question suivante sur l'assurance automobile au Québec. "
        "Fournissez uniquement la réponse, sans explication.\n\nQuestion : {question}\nRéponse :"
    ),
    "legal_simplification": (
        "Simplifiez le texte juridique suivant en langage clair, "
        "en préservant le sens légal.\n\nTexte : {question}\nSimplification :"
    ),
}


def get_prompt(question: str, domain: str) -> str:
    """Generate evaluation prompt."""
    template = DOMAIN_PROMPTS.get(domain, "Question: {question}\nAnswer:")
    return template.format(question=question)


# ---------------------------------------------------------------------------
# API callers — clients are cached to reuse HTTP connections
# ---------------------------------------------------------------------------

_clients: dict[str, object] = {}


def _get_client(provider: str):
    """Lazy-init and cache API clients."""
    if provider in _clients:
        return _clients[provider]

    if provider == "openai":
        from openai import OpenAI

        _clients[provider] = OpenAI()
    elif provider == "anthropic":
        from anthropic import Anthropic

        _clients[provider] = Anthropic()
    elif provider == "openrouter":
        from openai import OpenAI

        _clients[provider] = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
    elif provider == "mistral":
        from mistralai import Mistral

        _clients[provider] = Mistral(api_key=os.environ.get("MISTRAL_API_KEY", ""))
    elif provider == "google":
        import google.generativeai as genai

        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        _clients[provider] = genai.GenerativeModel("gemini-1.5-pro")
    return _clients[provider]


def call_openai(prompt: str, model_id: str) -> str:
    client = _get_client("openai")
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def call_anthropic(prompt: str, model_id: str) -> str:
    client = _get_client("anthropic")
    response = client.messages.create(
        model=model_id,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.content[0].text if response.content else "").strip()


def call_openrouter(prompt: str, model_id: str) -> str:
    client = _get_client("openrouter")
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def call_mistral(prompt: str, model_id: str) -> str:
    client = _get_client("mistral")
    response = client.chat.complete(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def call_google(prompt: str, model_id: str) -> str:
    import google.generativeai as genai

    # Cache per model_id since GenerativeModel is model-specific
    cache_key = f"google_{model_id}"
    if cache_key not in _clients:
        if "google" not in _clients:
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
            _clients["google"] = True
        _clients[cache_key] = genai.GenerativeModel(model_id)
    model = _clients[cache_key]
    response = model.generate_content(prompt)
    return (response.text or "").strip()


PROVIDERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "openrouter": call_openrouter,
    "mistral": call_mistral,
    "google": call_google,
}


# ---------------------------------------------------------------------------
# Scoring — compare prediction to reference
# ---------------------------------------------------------------------------


def _extract_number(text: str) -> float | None:
    """Extract the first number from text, handling $, %, commas."""
    if not text:
        return None
    # Remove markdown bold, dollar signs, commas, percent
    cleaned = re.sub(r"[*$,]", "", text)
    # Find first number (possibly negative, with decimals)
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if m:
        return float(m.group())
    return None


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, remove punctuation."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _extract_yes_no(text: str) -> str | None:
    """Extract yes/no from the beginning of a response.

    Uses word boundary to avoid false positives on "Not applicable",
    "None of the above", "Normal findings", etc.
    """
    t = text.strip().lower()
    m = re.match(r"^(yes|no)\b", t)
    if m:
        return m.group(1)
    return None


def _is_mcq_match(prediction: str, reference: str, options: dict | None) -> bool:
    """Check if MCQ answer matches (by option letter or text)."""
    pred_norm = _normalize_text(prediction)
    ref_norm = _normalize_text(reference)

    # Direct text match
    if ref_norm and ref_norm in pred_norm:
        return True

    # Single letter match (A, B, C, D)
    pred_letter = re.match(r"^([a-d])\b", pred_norm)
    ref_letter = re.match(r"^([a-d])\b", ref_norm)
    if pred_letter and ref_letter:
        return pred_letter.group(1) == ref_letter.group(1)

    return False


def score_prediction(
    prediction: str,
    reference: str,
    domain: str,
    options: dict | None = None,
    tolerance: float = 0.05,
) -> dict:
    """Score a prediction against a reference.

    Returns dict with:
        correct: bool
        score_method: str (exact, numeric, yes_no, mcq, fuzzy_contains)
    """
    if not prediction or not reference:
        return {"correct": False, "score_method": "empty"}

    # 1. MCQ (if options provided) — tested first because it's the most
    #    reliable signal: the dataset explicitly provides answer choices.
    if options and isinstance(options, dict):
        return {
            "correct": _is_mcq_match(prediction, reference, options),
            "score_method": "mcq",
        }

    # 2. Yes/No questions
    pred_yn = _extract_yes_no(prediction)
    ref_yn = _extract_yes_no(reference)
    if ref_yn is not None:
        return {
            "correct": pred_yn == ref_yn,
            "score_method": "yes_no",
        }

    # 3. Numeric comparison (with tolerance)
    ref_num = _extract_number(reference)
    pred_num = _extract_number(prediction)
    if ref_num is not None and pred_num is not None:
        if ref_num == 0:
            correct = abs(pred_num) < 0.01
        else:
            correct = abs(pred_num - ref_num) / abs(ref_num) <= tolerance
        return {"correct": correct, "score_method": "numeric"}

    # 4. Exact text match (normalized)
    pred_norm = _normalize_text(prediction)
    ref_norm = _normalize_text(reference)
    if pred_norm == ref_norm:
        return {"correct": True, "score_method": "exact"}

    # 5. Fuzzy: reference contained in prediction (min 4 chars to avoid
    #    false positives on very short references like "a", "no", "3m")
    if len(ref_norm) >= 4 and ref_norm in pred_norm:
        return {"correct": True, "score_method": "fuzzy_contains"}

    # 6. Partial: reference words all present in prediction (min 3 words)
    ref_words = set(ref_norm.split())
    pred_words = set(pred_norm.split())
    if len(ref_words) >= 3 and ref_words.issubset(pred_words):
        return {"correct": True, "score_method": "fuzzy_contains"}

    return {"correct": False, "score_method": "no_match"}


# ---------------------------------------------------------------------------
# wandb integration
# ---------------------------------------------------------------------------


def _init_wandb(args: argparse.Namespace) -> object | None:
    """Initialize wandb run if --wandb flag is set. Returns run or None."""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed, skipping logging. pip install wandb")
        return None

    run = wandb.init(
        project="severity-eval",
        config={
            "datasets": args.dataset,
            "models": args.model,
            "limit": args.limit,
            "delay": args.delay,
        },
        tags=["evaluation"],
    )
    return run


def _log_wandb_results(
    run,
    ds_name: str,
    model_name: str,
    results_df: pd.DataFrame,
) -> None:
    """Log evaluation results to wandb."""
    if run is None:
        return
    import wandb

    n_total = len(results_df)
    n_correct = results_df["correct"].sum()
    accuracy = n_correct / n_total if n_total > 0 else 0

    severity_counts = results_df["severity"].value_counts().to_dict()
    error_by_severity = results_df[~results_df["correct"]].groupby("severity").size().to_dict()
    score_methods = results_df["score_method"].value_counts().to_dict()

    metrics = {
        f"{ds_name}/{model_name}/accuracy": accuracy,
        f"{ds_name}/{model_name}/n_total": n_total,
        f"{ds_name}/{model_name}/n_correct": n_correct,
        f"{ds_name}/{model_name}/n_errors": n_total - n_correct,
        f"{ds_name}/{model_name}/error_rate": 1 - accuracy,
    }

    for sev, count in severity_counts.items():
        metrics[f"{ds_name}/{model_name}/severity_dist/{sev}"] = count
    for sev, count in error_by_severity.items():
        metrics[f"{ds_name}/{model_name}/errors_by_severity/{sev}"] = count
    for method, count in score_methods.items():
        metrics[f"{ds_name}/{model_name}/score_methods/{method}"] = count

    wandb.log(metrics)

    # Log results table
    table = wandb.Table(
        columns=["id", "severity", "question", "reference", "prediction", "correct", "score_method"],
        data=[
            [
                str(row["id"]),
                str(row["severity"]),
                str(row["question"])[:200],
                str(row["answer"])[:200],
                str(row["prediction"])[:200],
                bool(row["correct"]),
                str(row["score_method"]),
            ]
            for _, row in results_df.iterrows()
        ],
    )
    wandb.log({f"{ds_name}/{model_name}/results": table})


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_model(
    df: pd.DataFrame,
    model_name: str,
    output_path: Path,
    max_retries: int = 3,
    delay: float = 1.0,
) -> pd.DataFrame:
    """Evaluate a model on a dataset with scoring.

    Parameters
    ----------
    df : DataFrame
        Dataset with 'question', 'answer', 'severity', 'domain' columns.
    model_name : str
        Model key from MODELS dict.
    output_path : Path
        Path to save results.
    max_retries : int
        Max retries per query.
    delay : float
        Delay between API calls (seconds).

    Returns
    -------
    DataFrame with added 'prediction', 'correct', 'score_method' columns.
    """
    config = MODELS[model_name]
    call_fn = PROVIDERS[config["provider"]]
    model_id = config["model_id"]

    n_total = len(df)
    results = []
    n_errors_api = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        prompt = get_prompt(row["question"], row["domain"])

        prediction = ""
        for attempt in range(max_retries):
            try:
                prediction = call_fn(prompt, model_id)
                break
            except Exception as e:
                log.warning("Retry %d/%d for %s: %s", attempt + 1, max_retries, row["id"], e)
                time.sleep(delay * (attempt + 1))
        else:
            n_errors_api += 1
            log.error("All retries exhausted for %s", row["id"])

        # Score the prediction
        raw_opts = row.get("options") if "options" in row.index else None
        options = raw_opts if isinstance(raw_opts, dict) else None
        scoring = score_prediction(
            prediction,
            str(row["answer"]),
            row["domain"],
            options=options,
        )

        results.append(
            {
                **row.to_dict(),
                "model": model_name,
                "prediction": prediction,
                "correct": scoring["correct"],
                "score_method": scoring["score_method"],
            }
        )

        if delay > 0:
            time.sleep(delay)

        if (idx + 1) % 50 == 0:
            n_correct_so_far = sum(1 for r in results if r["correct"])
            log.info(
                "  [%d/%d] accuracy so far: %.1f%% (%d correct)",
                idx + 1,
                n_total,
                n_correct_so_far / len(results) * 100,
                n_correct_so_far,
            )

    results_df = pd.DataFrame(results)

    # Summary
    n_correct = results_df["correct"].sum()
    accuracy = n_correct / n_total if n_total > 0 else 0
    log.info(
        "  %s on %s: accuracy=%.1f%% (%d/%d), api_errors=%d",
        model_name,
        output_path.stem.split("_")[0],
        accuracy * 100,
        n_correct,
        n_total,
        n_errors_api,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_json(output_path, orient="records", indent=2, force_ascii=False)
    log.info("  Saved %d results to %s", len(results_df), output_path)

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on severity-annotated datasets")
    parser.add_argument(
        "--dataset",
        default="financebench",
        choices=list(DATASETS.keys()) + ["all"],
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        choices=list(MODELS.keys()) + ["all"],
    )
    parser.add_argument("--limit", type=int, default=None, help="Max instances per dataset")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (s)")
    parser.add_argument("--wandb", action="store_true", help="Log results to wandb")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    args = parser.parse_args()

    datasets_to_eval = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    models_to_eval = list(MODELS.keys()) if args.model == "all" else [args.model]

    # Validate API keys for selected models
    _PROVIDER_KEYS = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "google": "GEMINI_API_KEY",
    }
    needed_providers = {MODELS[m]["provider"] for m in models_to_eval}
    for provider in needed_providers:
        env_var = _PROVIDER_KEYS[provider]
        if not os.environ.get(env_var):
            log.error("Missing API key: %s (needed for %s)", env_var, provider)
            raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wandb_run = _init_wandb(args)

    for ds_name in datasets_to_eval:
        log.info("=" * 60)
        log.info("Dataset: %s", ds_name)
        log.info("=" * 60)

        # Load dataset
        module_path, func_name = DATASETS[ds_name].rsplit(":", 1)
        module = importlib.import_module(module_path)
        load_fn = getattr(module, func_name)
        df = load_fn(limit=args.limit)
        log.info("Loaded %d instances", len(df))
        log.info("Severity distribution:\n%s\n", df["severity"].value_counts().to_string())

        for model_name in models_to_eval:
            output_path = OUTPUT_DIR / f"{ds_name}_{model_name}.json"

            if output_path.exists() and not args.force:
                log.info("Results exist at %s, skipping (use --force to overwrite)", output_path)
                if wandb_run is not None:
                    existing_df = pd.read_json(output_path)
                    _log_wandb_results(wandb_run, ds_name, model_name, existing_df)
                continue

            log.info("Evaluating %s on %s...", model_name, ds_name)
            results_df = evaluate_model(df, model_name, output_path, delay=args.delay)
            _log_wandb_results(wandb_run, ds_name, model_name, results_df)

    if wandb_run is not None:
        wandb_run.finish()
        log.info("wandb run finished")


if __name__ == "__main__":
    main()
