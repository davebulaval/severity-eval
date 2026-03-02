#!/usr/bin/env bash
# =============================================================================
# run_local.sh — Run local models only via unsloth bnb-4bit (no API keys needed)
#
# Models: QwQ-32B, Qwen3-30B-A3B, DeepSeek-R1-Distill-14B, Qwen2.5-14B,
#         Phi-4, Prometheus-7B, Skywork-Critic-8B
#
# Usage:
#   ./experiments/run_local.sh                        # All local models
#   ./experiments/run_local.sh --wandb                # With wandb logging + artifacts
#   ./experiments/run_local.sh --limit 100            # Cap instances per dataset
#   ./experiments/run_local.sh --dry-run              # Show what would run
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Forward all arguments to run_all.sh with --skip-api
exec bash "$SCRIPT_DIR/run_all.sh" --skip-api "$@"
