#!/usr/bin/env bash
# End-to-end cross-modal VAE pipeline: .trk → .npz → train → generated .trk.
#
# Intended for PLGrid (or any GPU workstation): submit this single script to
# run data prep, training, and inference in one job. All four Python entry
# points are individually tested under tests/.
#
# Usage:
#   MRI_TRK=path/to/mri.trk PLI_TRK=path/to/pli.trk OUT_DIR=runs/exp1 \
#       DEVICE=cuda EPOCHS=200 scripts/run_full_pipeline.sh
#
# Required env vars:
#   MRI_TRK   path to MRI tractography .trk (from DSI Studio)
#   PLI_TRK   path to PLI tractography .trk (from run_pli_tractography.py)
#   OUT_DIR   output directory; all artifacts land here
#
# Optional env vars:
#   DEVICE        cuda | cpu             (default: cuda)
#   EPOCHS        training epochs        (default: 200)
#   K             bundle size            (default: 64)
#   N_POINTS      points per streamline  (default: 256)
#   BATCH_SIZE                           (default: 8)
#   BUNDLES_PER_EPOCH                    (default: 256)
#   SEED                                 (default: 0)

set -euo pipefail

MRI_TRK="data/processed/dti_MRI_streamlines_Sample1.trk"
PLI_TRK="data/processed/microscopy_tractography.trk"
OUT_DIR="runs/exp1"

: "${MRI_TRK:?MRI_TRK is required}"
: "${PLI_TRK:?PLI_TRK is required}"
: "${OUT_DIR:?OUT_DIR is required}"

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-200}"
K="${K:-64}"
N_POINTS="${N_POINTS:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BUNDLES_PER_EPOCH="${BUNDLES_PER_EPOCH:-256}"
SEED="${SEED:-0}"

mkdir -p "${OUT_DIR}"
RUN_DIR="${OUT_DIR}/run"
MRI_NPZ="${OUT_DIR}/mri_${N_POINTS}pts.npz"
PLI_NPZ="${OUT_DIR}/pli_${N_POINTS}pts.npz"
GEN_TRK="${OUT_DIR}/generated_pli.trk"

banner() {
    echo
    echo "=============================================================="
    echo "  $1"
    echo "=============================================================="
}

banner "Stage 1/4 — Resample MRI .trk → ${MRI_NPZ}"
uv run python -m cross_modal_vae.data.trk_to_npz \
    --trk "${MRI_TRK}" \
    --out "${MRI_NPZ}" \
    --n-points "${N_POINTS}"

banner "Stage 2/4 — Resample PLI .trk → ${PLI_NPZ}"
uv run python -m cross_modal_vae.data.trk_to_npz \
    --trk "${PLI_TRK}" \
    --out "${PLI_NPZ}" \
    --n-points "${N_POINTS}"

banner "Stage 3/4 — Train cross-modal VAE (epochs=${EPOCHS}, device=${DEVICE})"
uv run python -m cross_modal_vae.training.train \
    --mri "${MRI_NPZ}" \
    --pli "${PLI_NPZ}" \
    --out-dir "${RUN_DIR}" \
    --epochs "${EPOCHS}" \
    --K "${K}" \
    --n-points "${N_POINTS}" \
    --batch-size "${BATCH_SIZE}" \
    --bundles-per-epoch "${BUNDLES_PER_EPOCH}" \
    --seed "${SEED}" \
    --device "${DEVICE}"

banner "Stage 4/4 — Generate PLI-like streamlines from MRI → ${GEN_TRK}"
uv run python -m cross_modal_vae.evaluation.generate \
    --checkpoint "${RUN_DIR}/final.pt" \
    --mri "${MRI_NPZ}" \
    --stats-dir "${RUN_DIR}" \
    --out "${GEN_TRK}" \
    --K "${K}" \
    --n-points "${N_POINTS}" \
    --device "${DEVICE}"

banner "Done"
echo "Checkpoint:     ${RUN_DIR}/final.pt"
echo "Stats:          ${RUN_DIR}/stats_{mri,pli}.npz"
echo "History:        ${RUN_DIR}/history.json"
echo "Generated .trk: ${GEN_TRK}"
