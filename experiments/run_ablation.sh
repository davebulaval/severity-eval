#!/usr/bin/env bash
# =============================================================================
# run_ablation.sh — Run ablation study (HELM-style uniform prompts)
#
# Runs "standard" prompt configuration as an ablation comparing uniform
# prompts vs. published-paper prompts (run_experiments.sh).
#
# Results are saved with a _standard suffix:
#   experiments/results/{dataset}_{model}_standard.json
#
# Usage:
#   ./experiments/run_ablation.sh                    # Full matrix (wandb on)
#   ./experiments/run_ablation.sh --limit 100        # Cap instances per dataset
#   ./experiments/run_ablation.sh --dry-run           # Show what would run
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Forward all arguments to run_all.sh with --prompt-style standard
exec bash "$SCRIPT_DIR/run_all.sh" --prompt-style standard "$@"
