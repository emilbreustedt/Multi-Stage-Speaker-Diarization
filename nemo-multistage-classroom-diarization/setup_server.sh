#!/bin/bash
# ============================================================
# setup_server.sh  —  One-time setup on a Linux GPU server
#
# Requirements: Python 3.10, CUDA 12.x, NVIDIA GPU
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/4] Installing Python dependencies..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo ""
echo "[2/4] Cloning NeMo (if missing)..."
if [ ! -d "${SCRIPT_DIR}/NeMo" ]; then
    git clone https://github.com/NVIDIA/NeMo.git "${SCRIPT_DIR}/NeMo"
else
    echo "  NeMo already present, skipping."
fi

echo ""
echo "[3/4] Downloading wav2vec2 checkpoint..."
CKPT_DIR="${SCRIPT_DIR}/checkpoints/w2v2-robust-large-ckpt"
mkdir -p "${CKPT_DIR}"
if [ ! -f "${CKPT_DIR}/ckpt.pt" ]; then
    gdown --fuzzy 'https://drive.google.com/file/d/1f9mMqzpGaLA2RB0m7dcesxo4deOB_GDq/view?usp=sharing' \
          -O "${CKPT_DIR}/ckpt.pt"
else
    echo "  Checkpoint already present, skipping."
fi

echo ""
echo "[4/4] Creating output directories..."
mkdir -p "${SCRIPT_DIR}/vad_output_frames"
mkdir -p "${SCRIPT_DIR}/whisper_output_frames"
mkdir -p "${SCRIPT_DIR}/diarization_output/pred_rttms"

echo ""
echo "============================================================"
echo "  Setup complete. Run with:"
echo "  chmod +x run_my_data.sh && ./run_my_data.sh"
echo "============================================================"
