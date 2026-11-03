#!/usr/bin/env bash
# Run tests, the main experiments, and their figures with default settings.
# Run with bash symplectic_reck/run_all.sh [workers]   (parallel workers for finite-shot trials)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
workers="${1:-4}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -m unittest discover -s symplectic_reck/tests -v

python -u -m symplectic_reck.experiments.run_four_mode_nulling
python -m symplectic_reck.plotting.plot_four_mode_nulling

python -u -m symplectic_reck.experiments.run_conditioning
python -m symplectic_reck.plotting.plot_conditioning

python -u -m symplectic_reck.experiments.run_insitu_vs_tomog_with_gate_errors
python -m symplectic_reck.plotting.plot_insitu_vs_tomog_with_gate_errors

python -u -m symplectic_reck.experiments.run_finite_shots --max-workers "$workers"
python -m symplectic_reck.plotting.export_quartile_tables
python -m symplectic_reck.plotting.plot_finite_shots
python -m symplectic_reck.experiments.run_universal_architecture

python -m symplectic_reck.experiments.run_loss_simulations

echo "Finished. Results: symplectic_reck/results/"
