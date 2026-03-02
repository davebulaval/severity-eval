"""Pipeline d'évaluation des modèles via API avec scoring et logging wandb.

Usage:
    python -m experiments.evaluate_models --dataset financebench --model gpt-4o --limit 100
    python -m experiments.evaluate_models --dataset all --model all
    python -m experiments.evaluate_models --dataset financebench --model all --wandb
"""

from __future__ import annotations

import argparse
import importlib
import json as json_mod
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    # --- OpenAI ---
    "o3": {"provider": "openai", "model_id": "o3-2025-04-16"},
    "o3-mini": {"provider": "openai", "model_id": "o3-mini-2025-01-31"},
    "gpt-5.2": {"provider": "openai", "model_id": "gpt-5.2-2025-05-14"},
    # --- Anthropic ---
    "claude-opus": {"provider": "anthropic", "model_id": "claude-opus-4-6"},
    "claude-sonnet": {"provider": "anthropic", "model_id": "claude-sonnet-4-6"},
    "claude-haiku": {"provider": "anthropic", "model_id": "claude-haiku-4-5-20251001"},
    # --- Google ---
    "gemini-3.1-pro": {"provider": "google", "model_id": "gemini-3.1-pro-preview"},
    # --- xAI ---
    "grok-3": {"provider": "xai", "model_id": "grok-3-latest"},
    "grok-3-mini": {"provider": "xai", "model_id": "grok-3-mini-beta"},
    # --- Mistral ---
    "mistral-large": {"provider": "mistral", "model_id": "mistral-large-latest"},
    "mistral-medium": {"provider": "mistral", "model_id": "mistral-medium-2505"},
    # --- DeepSeek (via OpenRouter) ---
    "deepseek-r1": {"provider": "openrouter", "model_id": "deepseek/deepseek-r1"},
    "deepseek-v3": {"provider": "openrouter", "model_id": "deepseek/deepseek-chat-v3-0324"},
    # --- Qwen (via OpenRouter) ---
    "qwen3-235b-thinking": {"provider": "openrouter", "model_id": "qwen/qwen3-235b-a22b-thinking-2507"},
    "qwen3-235b": {"provider": "openrouter", "model_id": "qwen/qwen3-235b-a22b-2507"},
    # --- Local (unsloth bnb-4bit) ---
    "qwq-32b": {"provider": "local", "model_id": "unsloth/QwQ-32B-unsloth-bnb-4bit"},
    "qwen3-30b-a3b": {"provider": "local", "model_id": "Qwen/Qwen3-30B-A3B"},
    "gemma-2-27b": {"provider": "local", "model_id": "unsloth/gemma-2-27b-it-bnb-4bit"},
    "qwen3-14b": {"provider": "local", "model_id": "unsloth/Qwen3-14B-unsloth-bnb-4bit"},
    "deepseek-r1-distill-14b": {"provider": "local", "model_id": "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit"},
    "gemma-2-9b": {"provider": "local", "model_id": "unsloth/gemma-2-9b-it-bnb-4bit"},
    "granite-3.2-8b": {"provider": "local", "model_id": "unsloth/granite-3.2-8b-instruct-bnb-4bit"},
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
# Adaptive inference config per dataset: (max_new_tokens, max_length)
# ---------------------------------------------------------------------------

DATASET_INFERENCE_CONFIG: dict[str, tuple[int, int]] = {
    # dataset_name:     (max_new_tokens, max_length)
    "financebench":     (128, 4096),
    "finqa":            (128, 4096),
    "tatqa":            (128, 4096),
    "medcalc":          (64,  1024),
    "medqa":            (32,  1024),
    "headqa":           (32,  1024),
    "cuad":             (256, 4096),
    "maud":             (32,  4096),
    "contractnli":      (16,  2048),
    "rag_insurance":    (128, 2048),
    "judgebert":        (256, 2048),
}
_DEFAULT_INFERENCE_CONFIG = (128, 4096)


def _get_local_batch_size(model_id: str, max_length: int) -> int:
    """Compute optimal batch size based on model size tier and context length."""
    name = model_id.lower()
    if any(s in name for s in ["32b", "30b", "27b"]):
        size_tier = "large"    # ~15-18 GB VRAM
    elif any(s in name for s in ["14b"]):
        size_tier = "medium"   # ~8 GB VRAM
    else:
        size_tier = "small"    # ~5 GB VRAM (8B, 9B)

    if max_length >= 4096:
        return {"large": 2, "medium": 4, "small": 8}[size_tier]
    elif max_length >= 2048:
        return {"large": 4, "medium": 8, "small": 16}[size_tier]
    else:
        return {"large": 8, "medium": 16, "small": 32}[size_tier]

# ---------------------------------------------------------------------------
# Prompts — two styles: "original" (from published papers) and "standard"
# (uniform HELM-style). Selected via --prompt-style CLI flag.
#
# References for "original" prompts:
#   FinanceBench: Islam et al. 2023 (arXiv:2311.11944), Table 3 — Oracle format
#   FinQA: Chen et al. 2022 (Program-of-Thoughts, arXiv:2211.12588) — CoT format
#   TAT-QA: same structure as FinQA (table + paragraphs + question)
#   MedCalc-Bench: NCBI evaluation code — direct_answer mode
#   MedQA: lm-evaluation-harness (EleutherAI) — 0-shot MCQ format
#   HEAD-QA: MedHELM (Stanford CRFM) — 0-shot MCQ format
#   CUAD: LegalBench (Guha et al. 2023) — zero-shot jargon format
#   MAUD: LegalBench — MCQ classification format
#   ContractNLI: LegalBench — binarised NLI format
# ---------------------------------------------------------------------------

DATASET_PROMPTS: dict[str, dict[str, str]] = {
    # --- Finance ---
    "financebench": {
        # Islam et al. 2023, Oracle context-first (Table 3)
        # Output constraint added for automated evaluation (original paper used human eval)
        "original": (
            "Answer this question: {question}\n"
            "Provide only the answer, no explanation.\n"
            "Context:\n[START OF FILING] {context}[END OF FILING]\n"
            "Answer:"
        ),
        "standard": "Question: {question}\n{context}Answer:",
    },
    "finqa": {
        # Program-of-Thoughts CoT format (Chen et al. 2022)
        "original": (
            "Read the following text and table, and then answer a question:\n"
            "{context}"
            "Question: {question}\nAnswer:"
        ),
        "standard": "Question: {question}\n{context}Answer:",
    },
    "tatqa": {
        # Same structure as FinQA (table + passage context)
        "original": (
            "Read the following text and table, and then answer a question:\n"
            "{context}"
            "Question: {question}\nAnswer:"
        ),
        "standard": "Question: {question}\n{context}Answer:",
    },
    # --- Medical ---
    "medcalc": {
        # NCBI MedCalc-Bench evaluation code — direct_answer mode
        "original": (
            "You are a helpful assistant for calculating a score for a given "
            "patient note. Please output answer only without any other text.\n\n"
            "Here is the patient note:\n{question}\n\n"
            "Please output the numerical answer:"
        ),
        "standard": "Question: {question}\nAnswer:",
    },
    "medqa": {
        # lm-evaluation-harness (EleutherAI) — 0-shot MCQ format
        "original": "Question: {question}\n{options_str}Answer:",
        "standard": "Question: {question}\n{options_str}Answer:",
    },
    "headqa": {
        # MedHELM (Stanford CRFM) — 0-shot MCQ with biomedical instruction
        "original": (
            "You are a highly knowledgeable AI assistant specializing in biomedical "
            "sciences. Select the correct answer by outputting only the letter "
            "corresponding to your choice.\n\n"
            "Question: {question}\n{options_str}Answer:"
        ),
        "standard": "Question: {question}\n{options_str}Answer:",
    },
    # --- Legal ---
    "cuad": {
        # LegalBench (Guha et al. 2023) — zero-shot jargon, extractive QA
        "original": (
            "{context}"
            "Question: {question}\n"
            "Provide only the relevant text from the clause, or 'N/A' if not found.\nAnswer:"
        ),
        "standard": "Question: {question}\n{context}Answer:",
    },
    "maud": {
        # LegalBench — MCQ classification over merger agreement text
        "original": (
            "Instruction: Read the segment of a merger agreement and answer the "
            "multiple-choice question by choosing the option that best characterizes "
            "the agreement.\n"
            "Question: {question}\n{options_str}\n"
            "Merger Agreement: {context}\nAnswer:"
        ),
        "standard": "Question: {question}\n{options_str}{context}Answer:",
    },
    "contractnli": {
        # LegalBench — NLI classification (Entailment / Contradiction / Neutral)
        "original": (
            "Given the following NDA excerpt, determine the relationship with the "
            "hypothesis. Answer with exactly one word: Entailment, Contradiction, "
            "or Neutral.\n\n"
            "{context}"
            "Hypothesis: {question}\nAnswer:"
        ),
        "standard": (
            "Given the following text and hypothesis, answer with one word: "
            "Entailment, Contradiction, or Neutral.\n\n"
            "{context}"
            "Hypothesis: {question}\nAnswer:"
        ),
    },
    # --- Insurance (private) ---
    "rag_insurance": {
        "original": (
            "Répondez à la question suivante sur l'assurance automobile au Québec. "
            "Fournissez uniquement la réponse, sans explication.\n\n"
            "Question : {question}\nRéponse :"
        ),
        "standard": "Question : {question}\nRéponse :",
    },
    "judgebert": {
        "original": (
            "Simplifiez le texte juridique suivant en langage clair, "
            "en préservant le sens légal.\n\nTexte : {question}\nSimplification :"
        ),
        "standard": "Texte : {question}\nSimplification :",
    },
}

# Fallback for unknown datasets
_DEFAULT_PROMPT = "Question: {question}\n{context}{options_str}Answer:"


def get_prompt(
    question: str,
    dataset_name: str,
    evidence: str = "",
    options: dict | None = None,
    prompt_style: str = "original",
) -> str:
    """Generate evaluation prompt from dataset-specific templates.

    Parameters
    ----------
    question : str
    dataset_name : str
        Key in DATASET_PROMPTS (e.g. 'financebench', 'medqa').
    evidence : str
        Document context to inject.
    options : dict or None
        MCQ options {letter: text}.
    prompt_style : {'original', 'standard'}
        'original' uses prompts from published papers.
        'standard' uses uniform HELM-style template.
    """
    ds_prompts = DATASET_PROMPTS.get(dataset_name, {})
    template = ds_prompts.get(prompt_style, ds_prompts.get("original", _DEFAULT_PROMPT))

    context = f"Document excerpt:\n{evidence}\n\n" if evidence else ""
    options_str = ""
    if options and isinstance(options, dict):
        options_str = "\n".join(f"{k}. {v}" for k, v in sorted(options.items())) + "\n"

    # Use safe substitution to avoid crashes on curly braces in evidence/questions
    # (common in legal texts like CUAD/MAUD)
    try:
        return template.format(question=question, context=context, options_str=options_str)
    except (KeyError, ValueError, IndexError):
        # Fallback: manual replacement (handles stray { } in text)
        result = template.replace("{question}", question)
        result = result.replace("{context}", context)
        result = result.replace("{options_str}", options_str)
        return result


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
    elif provider == "xai":
        from openai import OpenAI

        _clients[provider] = OpenAI(
            base_url="https://api.x.ai/v1",
            api_key=os.environ.get("XAI_API_KEY", ""),
        )
    elif provider == "google":
        import google.generativeai as genai

        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        _clients[provider] = True  # sentinel — model created per model_id in call_google
    return _clients[provider]


def call_openai(prompt: str, model_id: str) -> str:
    client = _get_client("openai")
    # Reasoning models (o3, etc.) don't support temperature
    # and need more tokens (reasoning chain consumes most of the budget)
    is_reasoning = model_id.startswith("o3")
    kwargs = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
    }
    if is_reasoning:
        kwargs["max_completion_tokens"] = 8192
    else:
        kwargs["max_tokens"] = 512
        kwargs["temperature"] = 0.0
    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def call_anthropic(prompt: str, model_id: str) -> str:
    client = _get_client("anthropic")
    response = client.messages.create(
        model=model_id,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.content[0].text if response.content else "").strip()


def call_openrouter(prompt: str, model_id: str) -> str:
    client = _get_client("openrouter")
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def call_mistral(prompt: str, model_id: str) -> str:
    client = _get_client("mistral")
    response = client.chat.complete(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def call_xai(prompt: str, model_id: str) -> str:
    client = _get_client("xai")
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
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
    try:
        return (response.text or "").strip()
    except ValueError:
        # Response was blocked by safety filters
        return ""


PROVIDERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "openrouter": call_openrouter,
    "mistral": call_mistral,
    "xai": call_xai,
    "google": call_google,
}


# ---------------------------------------------------------------------------
# Scoring — compare prediction to reference
# ---------------------------------------------------------------------------


def _extract_number(text: str) -> float | None:
    """Extract the most relevant number from text.

    Priority order:
      1. Dollar amount ($X or $X million/billion)
      2. Percentage (X%)
      3. First standalone number (skipping years like 2018, 2019)
    """
    if not text:
        return None

    # 1. Dollar amounts — highest priority
    dollar_match = re.search(r"\$\s*([\d,.]+)\s*(billion|million|trillion)?", text, re.IGNORECASE)
    if dollar_match:
        num_str = dollar_match.group(1).replace(",", "")
        try:
            val = float(num_str)
            scale = (dollar_match.group(2) or "").lower()
            if scale == "trillion":
                val *= 1_000_000
            elif scale == "billion":
                val *= 1_000
            # million: keep as-is (most financial datasets use millions)
            return val
        except ValueError:
            pass

    # 2. Percentage
    pct_match = re.search(r"(-?[\d,.]+)\s*%", text)
    if pct_match:
        try:
            return float(pct_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # 3. First number that isn't a year (19xx/20xx)
    cleaned = re.sub(r"[*$,]", "", text)
    for m in re.finditer(r"-?\d+(?:\.\d+)?", cleaned):
        val = float(m.group())
        # Skip years
        if 1900 <= val <= 2099 and "." not in m.group():
            continue
        return val

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

    # Single letter match (A-Z — MAUD can have 5+ options)
    pred_letter = re.match(r"^([a-z])\b", pred_norm)
    ref_letter = re.match(r"^([a-z])\b", ref_norm)
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


def _upload_wandb_artifact(
    run,
    ds_name: str,
    model_name: str,
    output_path: Path,
) -> None:
    """Upload result JSON as a wandb Artifact for reproducibility."""
    if run is None or not output_path.exists():
        return
    import wandb

    artifact = wandb.Artifact(
        name=f"{ds_name}_{model_name}_results",
        type="evaluation_results",
        metadata={"dataset": ds_name, "model": model_name},
    )
    artifact.add_file(str(output_path))
    run.log_artifact(artifact)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def _build_prompt_for_row(
    row: pd.Series,
    dataset_name: str,
    prompt_style: str,
) -> tuple[str, dict | None]:
    """Build prompt and extract options for a single row."""
    evidence = row.get("evidence", "") if "evidence" in row.index else ""
    raw_opts = row.get("options") if "options" in row.index else None
    options = raw_opts if isinstance(raw_opts, dict) else None
    prompt = get_prompt(
        row["question"],
        dataset_name,
        evidence=str(evidence) if evidence else "",
        options=options,
        prompt_style=prompt_style,
    )
    return prompt, options


def _load_local_model(model_id: str):
    """Load a local model with unsloth (bnb-4bit) or HF AutoModel fallback.

    Unsloth models (prefixed 'unsloth/') use FastLanguageModel.
    Other models use AutoModelForCausalLM with BitsAndBytesConfig.
    """
    import torch

    cache_key = f"local_{model_id}"
    if cache_key in _clients:
        return _clients[cache_key]

    # Evict any previously loaded local model to free GPU memory
    for key in list(_clients):
        if key.startswith("local_") and key != cache_key:
            log.info("Unloading previous local model: %s", key)
            del _clients[key]
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if model_id.startswith("unsloth/"):
        from unsloth import FastLanguageModel

        log.info("Loading unsloth model %s ...", model_id)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_id,
            max_seq_length=4096,
            device_map="sequential",
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        model.eval()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        log.info("Loading HF model %s with bnb 4-bit ...", model_id)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            load_in_8bit=False,
            device_map="sequential",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model.eval()

    _clients[cache_key] = (model, tokenizer)
    return model, tokenizer


def _evaluate_local(
    df: pd.DataFrame,
    model_name: str,
    model_id: str,
    dataset_name: str,
    prompt_style: str,
    batch_size: int | None = None,
) -> list[dict]:
    """Batch inference for local models via unsloth/transformers."""
    import torch
    from transformers import pipeline

    # Adaptive inference config
    max_new_tokens, max_length = DATASET_INFERENCE_CONFIG.get(
        dataset_name, _DEFAULT_INFERENCE_CONFIG,
    )
    if batch_size is None:
        batch_size = _get_local_batch_size(model_id, max_length)
    log.info(
        "Local config: max_new_tokens=%d, max_length=%d, batch_size=%d",
        max_new_tokens, max_length, batch_size,
    )

    model, tokenizer = _load_local_model(model_id)

    # Build all prompts
    rows_data = []
    for _, row in df.iterrows():
        prompt, options = _build_prompt_for_row(row, dataset_name, prompt_style)
        # Apply chat template if tokenizer supports it
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted = prompt
        rows_data.append((row, formatted, options))

    # Create text-generation pipeline
    gen_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        dtype="float16",
        return_full_text=False,
        max_new_tokens=max_new_tokens,
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    # Batch inference
    from tqdm import tqdm

    all_prompts = [item[1] for item in rows_data]
    n_batches = (len(all_prompts) + batch_size - 1) // batch_size

    results = []
    pbar = tqdm(total=len(rows_data), desc=f"{model_name}", unit="sample")
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(rows_data))
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_rows = rows_data[batch_start:batch_end]

        start = time.perf_counter()
        with torch.no_grad():
            outputs = gen_pipeline(batch_prompts)
        elapsed = time.perf_counter() - start

        for (row, _prompt, options), output in zip(batch_rows, outputs, strict=True):
            prediction = output[0]["generated_text"].strip() if output else ""
            scoring = score_prediction(
                prediction, str(row["answer"]), row["domain"], options=options,
            )
            results.append({
                **row.to_dict(),
                "model": model_name,
                "prediction": prediction,
                "correct": scoring["correct"],
                "score_method": scoring["score_method"],
            })

        n_correct_so_far = sum(1 for r in results if r["correct"])
        acc = n_correct_so_far / len(results) * 100
        pbar.update(len(batch_prompts))
        pbar.set_postfix(acc=f"{acc:.1f}%", ms=f"{elapsed / len(outputs) * 1000:.0f}")

    pbar.close()
    return results


def _evaluate_batch_openai(
    df: pd.DataFrame,
    model_name: str,
    model_id: str,
    dataset_name: str,
    prompt_style: str,
    poll_interval: float = 30.0,
) -> list[dict]:
    """Batch inference via OpenAI Batch API (50% cost reduction)."""
    client = _get_client("openai")

    # Build JSONL requests
    rows_data: list[tuple[int, pd.Series, dict | None]] = []
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name
            for idx, (_, row) in enumerate(df.iterrows()):
                prompt, options = _build_prompt_for_row(row, dataset_name, prompt_style)
                rows_data.append((idx, row, options))

                is_reasoning = model_id.startswith("o3")
                body: dict = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if is_reasoning:
                    body["max_completion_tokens"] = 8192
                else:
                    body["max_tokens"] = 512
                    body["temperature"] = 0.0

                tmp.write(json_mod.dumps({
                    "custom_id": f"req-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }) + "\n")

        # Upload file (tmp is closed by 'with' block)
        log.info("  OpenAI Batch API: uploading %d requests...", len(rows_data))
        with open(tmp_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Create batch
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    log.info("  Batch created: %s", batch.id)

    # Poll until done
    while batch.status not in ("completed", "failed", "expired", "cancelled"):
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)
        counts = batch.request_counts
        completed = counts.completed if counts else 0
        total = counts.total if counts else len(rows_data)
        log.info("  Batch %s: %s (%d/%d completed)", batch.id, batch.status, completed, total)

    if batch.status != "completed":
        log.error("  Batch %s ended with status: %s", batch.id, batch.status)
        return [
            {**row.to_dict(), "model": model_name, "prediction": "",
             "correct": False, "score_method": "batch_error"}
            for _, row, _ in rows_data
        ]

    # Download and parse results
    result_content = client.files.content(batch.output_file_id)
    predictions: dict[int, str] = {}
    for line in result_content.text.strip().split("\n"):
        entry = json_mod.loads(line)
        req_idx = int(entry["custom_id"].split("-")[1])
        resp = entry.get("response", {})
        if resp.get("status_code") == 200:
            content = resp["body"]["choices"][0]["message"]["content"] or ""
            predictions[req_idx] = content.strip()
        else:
            predictions[req_idx] = ""

    # Score results
    results = []
    for idx, row, options in rows_data:
        prediction = predictions.get(idx, "")
        scoring = score_prediction(
            prediction, str(row["answer"]), row["domain"], options=options,
        )
        results.append({
            **row.to_dict(),
            "model": model_name,
            "prediction": prediction,
            "correct": scoring["correct"],
            "score_method": scoring["score_method"],
        })
    return results


def _evaluate_batch_anthropic(
    df: pd.DataFrame,
    model_name: str,
    model_id: str,
    dataset_name: str,
    prompt_style: str,
    poll_interval: float = 30.0,
) -> list[dict]:
    """Batch inference via Anthropic Message Batches API (50% cost reduction)."""
    client = _get_client("anthropic")

    # Build batch requests
    rows_data: list[tuple[int, pd.Series, dict | None]] = []
    batch_requests = []
    for idx, (_, row) in enumerate(df.iterrows()):
        prompt, options = _build_prompt_for_row(row, dataset_name, prompt_style)
        rows_data.append((idx, row, options))
        batch_requests.append({
            "custom_id": f"req-{idx}",
            "params": {
                "model": model_id,
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            },
        })

    log.info("  Anthropic Batch API: submitting %d requests...", len(batch_requests))
    message_batch = client.messages.batches.create(requests=batch_requests)
    log.info("  Batch created: %s", message_batch.id)

    # Poll until done
    while message_batch.processing_status != "ended":
        time.sleep(poll_interval)
        message_batch = client.messages.batches.retrieve(message_batch.id)
        counts = message_batch.request_counts
        done = (counts.succeeded + counts.errored) if counts else 0
        log.info("  Batch %s: %s (%d/%d done)", message_batch.id, message_batch.processing_status, done, len(rows_data))

    # Retrieve results
    predictions: dict[int, str] = {}
    for result in client.messages.batches.results(message_batch.id):
        req_idx = int(result.custom_id.split("-")[1])
        if result.result.type == "succeeded":
            msg = result.result.message
            content = msg.content[0].text if msg.content else ""
            predictions[req_idx] = content.strip()
        else:
            predictions[req_idx] = ""

    # Score results
    results = []
    for idx, row, options in rows_data:
        prediction = predictions.get(idx, "")
        scoring = score_prediction(
            prediction, str(row["answer"]), row["domain"], options=options,
        )
        results.append({
            **row.to_dict(),
            "model": model_name,
            "prediction": prediction,
            "correct": scoring["correct"],
            "score_method": scoring["score_method"],
        })
    return results


def _evaluate_batch_google(
    df: pd.DataFrame,
    model_name: str,
    model_id: str,
    dataset_name: str,
    prompt_style: str,
    poll_interval: float = 30.0,
) -> list[dict]:
    """Batch inference via Google Gemini Batch API (50% cost reduction)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

    # Build JSONL
    rows_data: list[tuple[int, pd.Series, dict | None]] = []
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name
            for idx, (_, row) in enumerate(df.iterrows()):
                prompt, options = _build_prompt_for_row(row, dataset_name, prompt_style)
                rows_data.append((idx, row, options))
                tmp.write(json_mod.dumps({
                    "key": f"req-{idx}",
                    "request": {
                        "contents": [{"parts": [{"text": prompt}]}],
                    },
                }) + "\n")

        log.info("  Google Batch API: uploading %d requests...", len(rows_data))
        uploaded_file = client.files.upload(
            file=tmp_path,
            config=types.UploadFileConfig(
                display_name=f"severity-eval-{dataset_name}-{model_name}",
                mime_type="jsonl",
            ),
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Create batch job
    batch_job = client.batches.create(
        model=model_id,
        src=uploaded_file.name,
        config={"display_name": f"severity-eval-{dataset_name}-{model_name}"},
    )
    log.info("  Batch created: %s", batch_job.name)

    # Poll
    completed_states = {
        "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
    }
    while batch_job.state.name not in completed_states:
        time.sleep(poll_interval)
        batch_job = client.batches.get(name=batch_job.name)
        log.info("  Batch %s: %s", batch_job.name, batch_job.state.name)

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        log.error("  Batch failed: %s", batch_job.state.name)
        return [
            {**row.to_dict(), "model": model_name, "prediction": "",
             "correct": False, "score_method": "batch_error"}
            for _, row, _ in rows_data
        ]

    # Download and parse results
    result_content = client.files.download(file=batch_job.dest.file_name)
    predictions: dict[int, str] = {}
    for line in result_content.decode("utf-8").strip().split("\n"):
        entry = json_mod.loads(line)
        req_idx = int(entry.get("key", "req-0").split("-")[1])
        resp = entry.get("response", {})
        candidates = resp.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            predictions[req_idx] = parts[0].get("text", "").strip() if parts else ""
        else:
            predictions[req_idx] = ""

    # Score results
    results = []
    for idx, row, options in rows_data:
        prediction = predictions.get(idx, "")
        scoring = score_prediction(
            prediction, str(row["answer"]), row["domain"], options=options,
        )
        results.append({
            **row.to_dict(),
            "model": model_name,
            "prediction": prediction,
            "correct": scoring["correct"],
            "score_method": scoring["score_method"],
        })
    return results


def _evaluate_batch_xai(
    df: pd.DataFrame,
    model_name: str,
    model_id: str,
    dataset_name: str,
    prompt_style: str,
    poll_interval: float = 30.0,
) -> list[dict]:
    """Batch inference via xAI Batch API (50% cost reduction)."""
    import httpx

    api_key = os.environ.get("XAI_API_KEY", "")
    base_url = "https://api.x.ai/v1"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Build requests
    rows_data: list[tuple[int, pd.Series, dict | None]] = []
    batch_requests = []
    for idx, (_, row) in enumerate(df.iterrows()):
        prompt, options = _build_prompt_for_row(row, dataset_name, prompt_style)
        rows_data.append((idx, row, options))
        batch_requests.append({
            "batch_request_id": f"req-{idx}",
            "batch_request": {
                "chat_get_completion": {
                    "messages": [{"role": "user", "content": prompt}],
                    "model": model_id,
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
            },
        })

    log.info("  xAI Batch API: submitting %d requests...", len(rows_data))

    with httpx.Client(timeout=120) as http:
        # Create batch
        resp = http.post(
            f"{base_url}/batches", headers=headers,
            json={"name": f"severity-eval-{dataset_name}-{model_name}"},
        )
        resp.raise_for_status()
        batch_id = resp.json()["batch_id"]
        log.info("  Batch created: %s", batch_id)

        # Add requests in chunks (rate limit: 100 calls / 30s)
        chunk_size = 100
        for i in range(0, len(batch_requests), chunk_size):
            chunk = batch_requests[i : i + chunk_size]
            resp = http.post(
                f"{base_url}/batches/{batch_id}/requests",
                headers=headers,
                json={"batch_requests": chunk},
            )
            resp.raise_for_status()
            if i + chunk_size < len(batch_requests):
                time.sleep(0.5)

        # Poll
        while True:
            time.sleep(poll_interval)
            resp = http.get(f"{base_url}/batches/{batch_id}", headers=headers)
            resp.raise_for_status()
            state = resp.json().get("state", {})
            pending = state.get("num_pending", 0)
            success = state.get("num_success", 0)
            total = state.get("num_requests", len(rows_data))
            log.info("  Batch %s: %d/%d done, %d pending", batch_id, success, total, pending)
            if pending == 0:
                break

        # Retrieve results (paginated)
        predictions: dict[int, str] = {}
        page_token = None
        while True:
            params: dict = {"page_size": 100}
            if page_token:
                params["pagination_token"] = page_token
            resp = http.get(
                f"{base_url}/batches/{batch_id}/results",
                headers=headers, params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            for result in data.get("succeeded", []):
                req_idx = int(result["batch_request_id"].split("-")[1])
                resp = result.get("response", {})
                # xAI batch returns chat completions structure
                choices = resp.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                else:
                    content = resp.get("content", "")  # fallback flat format
                predictions[req_idx] = content.strip()
            for result in data.get("failed", []):
                req_idx = int(result["batch_request_id"].split("-")[1])
                predictions[req_idx] = ""
            page_token = data.get("pagination_token")
            if not page_token:
                break

    # Score results
    results = []
    for idx, row, options in rows_data:
        prediction = predictions.get(idx, "")
        scoring = score_prediction(
            prediction, str(row["answer"]), row["domain"], options=options,
        )
        results.append({
            **row.to_dict(),
            "model": model_name,
            "prediction": prediction,
            "correct": scoring["correct"],
            "score_method": scoring["score_method"],
        })
    return results


def _evaluate_batch_mistral(
    df: pd.DataFrame,
    model_name: str,
    model_id: str,
    dataset_name: str,
    prompt_style: str,
    poll_interval: float = 30.0,
) -> list[dict]:
    """Batch inference via Mistral Batch API (50% cost reduction)."""
    client = _get_client("mistral")

    # Build JSONL
    rows_data: list[tuple[int, pd.Series, dict | None]] = []
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name
            for idx, (_, row) in enumerate(df.iterrows()):
                prompt, options = _build_prompt_for_row(row, dataset_name, prompt_style)
                rows_data.append((idx, row, options))
                tmp.write(json_mod.dumps({
                    "custom_id": f"req-{idx}",
                    "body": {
                        "max_tokens": 512,
                        "temperature": 0.0,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                }) + "\n")

        log.info("  Mistral Batch API: uploading %d requests...", len(rows_data))
        with open(tmp_path, "rb") as f:
            batch_data = client.files.upload(
                file={"file_name": "severity-eval-batch.jsonl", "content": f},
                purpose="batch",
            )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Create batch job
    created_job = client.batch.jobs.create(
        input_files=[batch_data.id],
        model=model_id,
        endpoint="/v1/chat/completions",
        metadata={"dataset": dataset_name, "model": model_name},
    )
    log.info("  Batch created: %s", created_job.id)

    # Poll
    terminal_states = {"SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"}
    while created_job.status not in terminal_states:
        time.sleep(poll_interval)
        created_job = client.batch.jobs.get(job_id=created_job.id)
        log.info("  Batch %s: %s", created_job.id, created_job.status)

    if created_job.status != "SUCCESS":
        log.error("  Batch failed: %s", created_job.status)
        return [
            {**row.to_dict(), "model": model_name, "prediction": "",
             "correct": False, "score_method": "batch_error"}
            for _, row, _ in rows_data
        ]

    # Download and parse results
    output_stream = client.files.download(file_id=created_job.output_file)
    predictions: dict[int, str] = {}
    for line in output_stream.read().decode("utf-8").strip().split("\n"):
        entry = json_mod.loads(line)
        req_idx = int(entry["custom_id"].split("-")[1])
        resp = entry.get("response", {})
        if resp.get("status_code") == 200:
            content = resp["body"]["choices"][0]["message"]["content"] or ""
            predictions[req_idx] = content.strip()
        else:
            predictions[req_idx] = ""

    # Score results
    results = []
    for idx, row, options in rows_data:
        prediction = predictions.get(idx, "")
        scoring = score_prediction(
            prediction, str(row["answer"]), row["domain"], options=options,
        )
        results.append({
            **row.to_dict(),
            "model": model_name,
            "prediction": prediction,
            "correct": scoring["correct"],
            "score_method": scoring["score_method"],
        })
    return results


def _score_and_log_results(
    results: list[dict],
    model_name: str,
    dataset_name: str,
    output_path: Path,
    n_total: int,
) -> pd.DataFrame:
    """Score, log, and save evaluation results."""
    results_df = pd.DataFrame(results)
    n_correct = results_df["correct"].sum()
    accuracy = n_correct / n_total if n_total > 0 else 0
    log.info(
        "  %s on %s: accuracy=%.1f%% (%d/%d)",
        model_name, dataset_name, accuracy * 100, n_correct, n_total,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_json(output_path, orient="records", indent=2, force_ascii=False)
    log.info("  Saved %d results to %s", len(results_df), output_path)
    return results_df


def evaluate_model(
    df: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    output_path: Path,
    max_retries: int = 3,
    delay: float = 1.0,
    prompt_style: str = "original",
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Evaluate a model on a dataset with scoring.

    Parameters
    ----------
    df : DataFrame
        Dataset with 'question', 'answer', 'severity', 'domain' columns.
    model_name : str
        Model key from MODELS dict.
    dataset_name : str
        Key in DATASET_PROMPTS (e.g. 'financebench', 'medqa').
    output_path : Path
        Path to save results.
    max_retries : int
        Max retries per query.
    delay : float
        Delay between API calls (seconds).
    prompt_style : {'original', 'standard'}
        Prompt template variant. 'original' uses published paper prompts.
    batch_size : int | None
        Batch size for local inference / max parallel API workers.
        None = auto (adaptive for local, 32 for API).

    Returns
    -------
    DataFrame with added 'prediction', 'correct', 'score_method' columns.
    """
    config = MODELS[model_name]
    provider = config["provider"]
    model_id = config["model_id"]
    n_total = len(df)

    # --- Dispatch by provider ---
    # Batch APIs (50% cost reduction): OpenAI, Anthropic, Google, xAI, Mistral
    # ThreadPoolExecutor fallback: OpenRouter (no batch API)

    _BATCH_DISPATCHERS = {
        "local": lambda: _evaluate_local(
            df, model_name, model_id, dataset_name, prompt_style, batch_size,
        ),
        "openai": lambda: _evaluate_batch_openai(
            df, model_name, model_id, dataset_name, prompt_style,
        ),
        "anthropic": lambda: _evaluate_batch_anthropic(
            df, model_name, model_id, dataset_name, prompt_style,
        ),
        "google": lambda: _evaluate_batch_google(
            df, model_name, model_id, dataset_name, prompt_style,
        ),
        "xai": lambda: _evaluate_batch_xai(
            df, model_name, model_id, dataset_name, prompt_style,
        ),
        "mistral": lambda: _evaluate_batch_mistral(
            df, model_name, model_id, dataset_name, prompt_style,
        ),
    }

    if provider in _BATCH_DISPATCHERS:
        results = _BATCH_DISPATCHERS[provider]()
        return _score_and_log_results(results, model_name, dataset_name, output_path, n_total)

    # Fallback: parallel requests with ThreadPoolExecutor (OpenRouter)
    call_fn = PROVIDERS[provider]

    def _call_single(idx_row):
        idx, row = idx_row
        prompt, options = _build_prompt_for_row(row, dataset_name, prompt_style)
        prediction = ""
        for attempt in range(max_retries):
            try:
                prediction = call_fn(prompt, model_id)
                break
            except Exception as e:
                log.warning("Retry %d/%d for %s: %s", attempt + 1, max_retries, row["id"], e)
                time.sleep(delay * (attempt + 1))
        else:
            log.error("All %d retries exhausted for %s", max_retries, row["id"])
        scoring = score_prediction(
            prediction, str(row["answer"]), row["domain"], options=options,
        )
        return idx, {
            **row.to_dict(),
            "model": model_name,
            "prediction": prediction,
            "correct": scoring["correct"],
            "score_method": scoring["score_method"],
        }

    results_indexed: list[dict | None] = [None] * n_total
    n_done = 0
    workers = min(batch_size or 32, n_total)
    checkpoint_interval = 50  # Save partial results every N completions
    checkpoint_path = output_path.with_suffix(".partial.json")

    # Resume from partial results if they exist
    if checkpoint_path.exists():
        try:
            partial_df = pd.read_json(checkpoint_path)
            for _, row_data in partial_df.iterrows():
                row_dict = row_data.to_dict()
                # Find the index by matching id
                for i, (_, orig_row) in enumerate(df.iterrows()):
                    if str(orig_row["id"]) == str(row_dict.get("id")):
                        results_indexed[i] = row_dict
                        n_done += 1
                        break
            log.info("  Resumed %d/%d results from %s", n_done, n_total, checkpoint_path)
        except Exception as e:
            log.warning("  Could not resume from checkpoint: %s", e)
            n_done = 0
            results_indexed = [None] * n_total

    log.info("  API parallel inference: %d prompts with %d workers", n_total, workers)

    # Only submit tasks for indices not yet completed
    pending_items = [
        (idx, row)
        for idx, (_, row) in enumerate(df.iterrows())
        if results_indexed[idx] is None
    ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_call_single, (idx, row)): idx
            for idx, row in pending_items
        }
        for future in as_completed(futures):
            try:
                idx, result = future.result()
            except Exception as e:
                idx = futures[future]
                log.error("Unhandled error for index %d: %s", idx, e)
                # Retrieve the row to preserve all columns in the DataFrame
                row = df.iloc[idx]
                result = {
                    **row.to_dict(),
                    "model": model_name, "prediction": "",
                    "correct": False, "score_method": "error",
                }
            results_indexed[idx] = result
            n_done += 1
            if n_done % checkpoint_interval == 0:
                n_correct = sum(1 for r in results_indexed if r and r.get("correct"))
                log.info(
                    "  [%d/%d] accuracy so far: %.1f%%",
                    n_done, n_total, n_correct / n_done * 100,
                )
                # Incremental checkpoint save
                completed = [r for r in results_indexed if r is not None]
                pd.DataFrame(completed).to_json(
                    checkpoint_path, orient="records", indent=2, force_ascii=False,
                )

    # Clean up checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Fill any remaining None entries (shouldn't happen, but defensive)
    for i, r in enumerate(results_indexed):
        if r is None:
            results_indexed[i] = {"model": model_name, "prediction": "", "correct": False, "score_method": "missing"}

    return _score_and_log_results(results_indexed, model_name, dataset_name, output_path, n_total)


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on severity-annotated datasets")
    parser.add_argument(
        "--dataset",
        default="financebench",
        choices=list(DATASETS.keys()) + ["all"],
    )
    parser.add_argument(
        "--model",
        default="o3",
        choices=list(MODELS.keys()) + ["all"],
    )
    parser.add_argument("--limit", type=int, default=None, help="Max instances per dataset")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between API calls (s)")
    parser.add_argument(
        "--prompt-style",
        default="original",
        choices=["original", "standard"],
        help="Prompt style: 'original' (from papers) or 'standard' (uniform HELM-style)",
    )
    parser.add_argument("--wandb", action="store_true", help="Log results to wandb")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="CUDA device ID(s) for local models (e.g. '0', '1', '0,1')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for local inference / max parallel API workers (auto if omitted)",
    )
    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES before any GPU library import
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        log.info("CUDA_VISIBLE_DEVICES=%s", args.gpu)

    datasets_to_eval = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    models_to_eval = list(MODELS.keys()) if args.model == "all" else [args.model]

    # Validate API keys for selected models (local is GPU, no key needed)
    _PROVIDER_KEYS = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "xai": "XAI_API_KEY",
        "google": "GEMINI_API_KEY",
    }
    needed_providers = {MODELS[m]["provider"] for m in models_to_eval}
    for provider in needed_providers:
        if provider == "local":
            continue
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
            suffix = f"_{args.prompt_style}" if args.prompt_style != "original" else ""
            output_path = OUTPUT_DIR / f"{ds_name}_{model_name}{suffix}.json"

            if output_path.exists() and not args.force:
                log.info("Results exist at %s, skipping (use --force to overwrite)", output_path)
                if wandb_run is not None:
                    existing_df = pd.read_json(output_path)
                    _log_wandb_results(wandb_run, ds_name, model_name, existing_df)
                    _upload_wandb_artifact(wandb_run, ds_name, model_name, output_path)
                continue

            log.info("Evaluating %s on %s (prompt=%s)...", model_name, ds_name, args.prompt_style)
            results_df = evaluate_model(
                df, model_name, ds_name, output_path,
                delay=args.delay, prompt_style=args.prompt_style,
                batch_size=args.batch_size,
            )
            _log_wandb_results(wandb_run, ds_name, model_name, results_df)
            _upload_wandb_artifact(wandb_run, ds_name, model_name, output_path)

    if wandb_run is not None:
        wandb_run.finish()
        log.info("wandb run finished")


if __name__ == "__main__":
    main()
