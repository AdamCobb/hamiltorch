#!/usr/bin/env bash
# run_experiments.sh — Run all surrogate HMC/RMHMC experiments and generate figures.
#
# Usage:
#   HAMILTORCH_GPU=<idx> bash run_experiments.sh [--device <cpu|cuda|mps:0>]
#                           [--smoke] [--skip-analytic] [--skip-rmhmc]
#                           [--skip-gp] [--no-container]
#
# HAMILTORCH_GPU selects which physical GPU to attach (default 0). Pin this when
# another job is using the machine, otherwise wall-clock timings — and hence
# every ESS/s number — are contaminated by contention.
#
# By default, when this script is invoked on the host and the
# localhost/approximate-hamiltorch container image is available, it re-executes
# itself inside that image with the GPU attached (matching .devcontainer).
# Use --no-container to run directly on the host environment.
#
# Outputs (all written to experiments/):
#   diagnostic_results.csv          — main leapfrog experiment
#   diagnostic_results_analytic.csv — analytic Gaussian experiment
#   snn_gradient_ablation.csv       — SNN gradient ablation
#   rmhmc_results.csv               — RMHMC surrogate experiment
#   diagnostic_results_gp.csv       — GP regression (expensive likelihood)
#   ess_vs_training_size.png        — sample-efficiency figure for paper
#   <distribution>_samples.png / <distribution>_reversibility.png
#   ess_stats.png / reversibility_stats.png / hamiltonian_conservation.png

set -euo pipefail

# ── Argument parsing ────────────────────────────────────────────────────────
DEVICE=""
SKIP_ANALYTIC=0
SKIP_RMHMC=0
SKIP_GP=0
SMOKE=0
NO_CONTAINER=0
IN_CONTAINER="${HAMILTORCH_IN_CONTAINER:-0}"

ARGS=("$@")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)        DEVICE="$2"; shift 2 ;;
        --skip-analytic) SKIP_ANALYTIC=1; shift ;;
        --skip-rmhmc)    SKIP_RMHMC=1; shift ;;
        --skip-gp)       SKIP_GP=1; shift ;;
        --smoke)         SMOKE=1; shift ;;
        --no-container)  NO_CONTAINER=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Re-exec inside the GPU container when running on the host ───────────────
CONTAINER_IMAGE="localhost/approximate-hamiltorch:latest"
if [[ "$IN_CONTAINER" -eq 0 && "$NO_CONTAINER" -eq 0 ]] \
   && command -v podman >/dev/null 2>&1 \
   && podman image exists "$CONTAINER_IMAGE" 2>/dev/null; then
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "Re-executing inside $CONTAINER_IMAGE with GPU attached..."
    exec podman run --rm \
        --user=1000:100 \
        --userns=keep-id:uid=1000,gid=100 \
        --device=nvidia.com/gpu="${HAMILTORCH_GPU:-0}" \
        --security-opt=label=disable \
        -e HAMILTORCH_IN_CONTAINER=1 \
        -v "$REPO_DIR":/home/jovyan/work:z \
        -w /home/jovyan/work \
        "$CONTAINER_IMAGE" \
        bash run_experiments.sh "${ARGS[@]}"
fi

# ── Ensure python dependencies not baked into the image ─────────────────────
python3 -c "import termcolor, torchdyn, arviz" 2>/dev/null || \
    pip install --quiet termcolor torchdyn arviz

# ── Auto-detect device if not specified ─────────────────────────────────────
if [[ -z "$DEVICE" ]]; then
    DEVICE=$(python3 -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif torch.backends.mps.is_available():
    print('mps:0')
else:
    print('cpu')
")
fi
echo "Using device: $DEVICE"

SMOKE_FLAG=""
if [[ $SMOKE -eq 1 ]]; then
    SMOKE_FLAG="--smoke"
    echo "Smoke mode: tiny chains/epochs, pipeline validation only."
fi

# ── Ensure output directories exist ─────────────────────────────────────────
mkdir -p experiments
mkdir -p logs

# ── Helper: timestamped log prefix ──────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. Main leapfrog experiment (all distributions, full training-size sweep)
log "Starting main leapfrog experiment..."
python3 main.py --experiment sample_size --device "$DEVICE" $SMOKE_FLAG \
    2>&1 | tee logs/sample_size.log
log "Main leapfrog experiment complete."

# ── 2. Analytic Gaussian experiment (high-dimensional / warped Gaussian) ────
if [[ $SKIP_ANALYTIC -eq 0 ]]; then
    log "Starting analytic Gaussian experiment..."
    python3 main.py --experiment sample_size_analytic --device "$DEVICE" $SMOKE_FLAG \
        2>&1 | tee logs/sample_size_analytic.log
    log "Analytic experiment complete."
else
    log "Skipping analytic experiment (--skip-analytic set)."
fi

# ── 3. SNN gradient ablation ────────────────────────────────────────────────
log "Starting SNN gradient ablation..."
python3 main.py --experiment snn_ablation --device "$DEVICE" $SMOKE_FLAG \
    2>&1 | tee logs/snn_ablation.log
log "SNN gradient ablation complete."

# ── 4. GP regression experiment (expensive likelihood) ──────────────────────
if [[ $SKIP_GP -eq 0 ]]; then
    log "Starting GP regression experiment..."
    python3 main.py --experiment gp --device "$DEVICE" $SMOKE_FLAG \
        2>&1 | tee logs/gp.log
    log "GP experiment complete."
else
    log "Skipping GP experiment (--skip-gp set)."
fi

# ── 5. RMHMC surrogate experiment ───────────────────────────────────────────
if [[ $SKIP_RMHMC -eq 0 ]]; then
    log "Starting RMHMC surrogate experiment..."
    python3 main.py --experiment rmhmc --device "$DEVICE" $SMOKE_FLAG \
        2>&1 | tee logs/rmhmc.log
    log "RMHMC experiment complete."
else
    log "Skipping RMHMC experiment (--skip-rmhmc set)."
fi

# ── 6. Regenerate ESS vs training-size figure from latest results ───────────
log "Generating ESS vs training-size figure..."
python3 main.py --experiment plot_ess --device cpu \
    2>&1 | tee logs/plot_ess.log
log "Figure saved to experiments/ess_vs_training_size.png"

# ── Done ────────────────────────────────────────────────────────────────────
log "All experiments finished. Results in experiments/, logs in logs/."
