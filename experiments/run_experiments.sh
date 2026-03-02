#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — Run primary experiments for the EMNLP paper
#
# Runs "original" prompt configuration (prompts from published papers).
# For ablation study (HELM-style uniform prompts), see run_ablation.sh.
#
# Usage:
#   ./experiments/run_experiments.sh                    # Full matrix (wandb on)
#   ./experiments/run_experiments.sh --limit 100        # Cap instances per dataset
#   ./experiments/run_experiments.sh --dry-run           # Show what would run
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Forward all arguments to run_all.sh with --prompt-style original
exec bash "$SCRIPT_DIR/run_all.sh" --prompt-style original "$@"
