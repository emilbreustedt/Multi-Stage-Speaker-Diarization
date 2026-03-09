#!/bin/bash
# ============================================================
# run_my_data.sh  —  Diarization pipeline for HIEB classroom data
#
# Requirements:
#   - NVIDIA GPU + CUDA
#   - pip install -r requirements.txt  (Python 3.10 recommended)
#   - Checkpoint: checkpoints/w2v2-robust-large-ckpt/ckpt.pt
#     Download: gdown --fuzzy 'https://drive.google.com/file/d/1f9mMqzpGaLA2RB0m7dcesxo4deOB_GDq/view?usp=sharing' -O checkpoints/w2v2-robust-large-ckpt/ckpt.pt
#
# Usage:
#   cd nemo-multistage-classroom-diarization
#   chmod +x run_my_data.sh
#   ./run_my_data.sh
# ============================================================

set -e  # stop on any error

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT="${SCRIPT_DIR}/checkpoints/w2v2-robust-large-ckpt/ckpt.pt"
MANIFEST_FILE="${SCRIPT_DIR}/manifests/my_data.json"

VAD_OUTPUT="${SCRIPT_DIR}/vad_output_frames"
WHISPER_OUTPUT="${SCRIPT_DIR}/whisper_output_frames"
DIAR_OUTPUT_DIR="${SCRIPT_DIR}/diarization_output"
RESULTS_FILE="${SCRIPT_DIR}/DER_results_my_data.txt"
DIARIZER_LOG="${SCRIPT_DIR}/diarizer_temp.log"

# ── Preflight checks ────────────────────────────────────────
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Checkpoint not found at ${CKPT}"
    echo "Run: gdown --fuzzy 'https://drive.google.com/file/d/1f9mMqzpGaLA2RB0m7dcesxo4deOB_GDq/view?usp=sharing' -O ${CKPT}"
    exit 1
fi

if [ ! -f "${MANIFEST_FILE}" ]; then
    echo "ERROR: Manifest not found at ${MANIFEST_FILE}"
    echo "Run prepare_data.ipynb first."
    exit 1
fi

echo "============================================================"
echo "  Manifest : ${MANIFEST_FILE}"
echo "  Checkpoint: ${CKPT}"
echo "  Output    : ${DIAR_OUTPUT_DIR}"
echo "============================================================"

# ── Stage 1: VAD frames (wav2vec2) ──────────────────────────
echo ""
echo "[Stage 1] Generating VAD frames..."
python "${SCRIPT_DIR}/generate_w2v2_speech_labels/run_vad.py" \
    --manifest_file "${MANIFEST_FILE}" \
    --checkpoint_path "${CKPT}" \
    --vad_manifest_path "null.json" \
    --frames_output_path "${VAD_OUTPUT}"

# ── Stage 2: Whisper frames + save transcripts for WER ──────
echo ""
echo "[Stage 2] Running Whisper transcription..."
python "${SCRIPT_DIR}/generate_whisper_speech_labels/whisper_transcribe.py" \
    --manifest_file "${MANIFEST_FILE}" \
    --output_dir "${WHISPER_OUTPUT}"

# ── Stage 3: Parameter sweep + diarization ──────────────────
echo ""
echo "[Stage 3] Parameter sweep + diarization..."
rm -f "${RESULTS_FILE}"

for alpha in $(seq 0.20 0.20 1.0)
do
  for offset in $(seq 0.10 0.05 0.80)
  do
    for onset in $(seq 0.30 0.05 0.90)
    do
      if (( $(echo "$onset > $offset" | bc -l) )); then
        echo "--------------------------------------------------"
        echo "alpha=${alpha}, onset=${onset}, offset=${offset}"
        echo "--------------------------------------------------"

        rm -f "${SCRIPT_DIR}/vad_outs.json"
        rm -rf pymp-* tmp* torchelastic*

        # Combine VAD + Whisper frames
        python "${SCRIPT_DIR}/run_diarization/tune_vad_params.py" \
            --manifest_file="${MANIFEST_FILE}" \
            --frame_dir="${VAD_OUTPUT}" \
            --asr_dir="${WHISPER_OUTPUT}" \
            --alpha="${alpha}" \
            --onset="${onset}" \
            --offset="${offset}" \
            --out_dir="${SCRIPT_DIR}/vad_outs.json"

        # Run NeMo diarizer
        python3 "${SCRIPT_DIR}/NeMo/examples/speaker_tasks/diarization/clustering_diarizer/offline_diar_infer.py" \
            diarizer.manifest_filepath="${MANIFEST_FILE}" \
            diarizer.out_dir="${DIAR_OUTPUT_DIR}" \
            diarizer.vad.model_path=null \
            diarizer.vad.external_vad_manifest="${SCRIPT_DIR}/vad_outs.json" \
            diarizer.speaker_embeddings.parameters.save_embeddings=False \
            diarizer.speaker_embeddings.model_path='titanet_large' \
            diarizer.clustering.parameters.oracle_num_speakers=True \
            2>&1 | tee "${DIARIZER_LOG}"

        FA_LINE=$(grep '| FA' "${DIARIZER_LOG}" || true)
        if [ -n "${FA_LINE}" ]; then
            echo "alpha=${alpha}, onset=${onset}, offset=${offset}, ${FA_LINE}" >> "${RESULTS_FILE}"
        else
            echo "alpha=${alpha}, onset=${onset}, offset=${offset}, NO_FA_LINE_FOUND" >> "${RESULTS_FILE}"
        fi
      fi
    done
  done
done

echo ""
echo "============================================================"
echo "  Done. Results: ${RESULTS_FILE}"
echo "  Best params:"
sort -t'|' -k5 -n "${RESULTS_FILE}" | head -3
echo "============================================================"

# Find best parameter combination
python "${SCRIPT_DIR}/find_best_combination.py"
