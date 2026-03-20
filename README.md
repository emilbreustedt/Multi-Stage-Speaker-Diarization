# Multi-Stage Speaker Diarization for Classroom Audio

Speaker diarization of German-language classroom recordings using two independent pipelines. Both pipelines identify *who spoke when* and classify speakers as **Teacher** or **Student**.

The audio data consists of ~141 stereo WAV sessions (lectures and small-group tutoring) recorded in Swiss classrooms.

---

## Pipelines

### 1. NeMo Multi-Stage (`nemo-multistage-classroom-diarization/`)

Based on the paper [Multi-Stage Speaker Diarization for Noisy Classrooms](literature/2505.10879v2.pdf) (Khan et al., EDM 2025). The `nemo-multistage-classroom-diarization/` folder is an adapted version of their [original repository](https://github.com/EduNLP/edm25-nemo-classroom-diarization), extended to run on Windows with our HIEB classroom dataset.

Combines a fine-tuned **wav2vec2-robust-large** VAD model with **Whisper** speech labels, then runs **NeMo TitaNet-Large** speaker clustering.

**Stages:**
1. **VAD frames** — wav2vec2-robust-large checkpoint produces per-frame speech probabilities
2. **Whisper frames** — Whisper large-v2 provides word-level speech labels and full transcripts
3. **Frame fusion + diarization** — frames are combined as `combined = vad + α × whisper`, thresholded with hysteresis (onset/offset), then fed to NeMo's clustering diarizer
4. **Parameter sweep** — grid search over α ∈ {0.2, 0.6, 1.0}, onset ∈ {0.3–0.9}, offset ∈ {0.1–0.7}

**Results:** DER ~30% on test sessions; speaker count accuracy 100% (oracle num_speakers).

See [`nemo-multistage-classroom-diarization/README.md`](nemo-multistage-classroom-diarization/README.md) for full setup details.

---

### 2. Pyannote + Whisper (`pyannote-whisper-diarization/`)

Runs **pyannote.audio 4.x** speaker diarization followed by **Whisper large-v2** transcription, then aligns transcript segments to speaker turns. Teacher/Student roles are assigned by heuristic scoring (speaking time, segment length, question rate, etc.).

**Results:** 70.46% average sequence match across all sessions (Teacher/Student turn alternation).

---

## Repository Structure

```
.
├── nemo-multistage-classroom-diarization/
│   ├── generate_w2v2_speech_labels/   # Stage 1: wav2vec2 VAD
│   ├── generate_whisper_speech_labels/# Stage 2: Whisper frame labels + transcripts
│   ├── run_diarization/               # Stage 3: VAD fusion
│   ├── ground_truth/rttm/             # Reference RTTMs (141 sessions)
│   ├── diarization_output/pred_rttms/ # NeMo diarization output
│   ├── DER_results_my_data.txt        # Parameter sweep results
│   ├── find_best_combination.py       # Parse best α/onset/offset from sweep
│   ├── run_my_data.sh                 # Main pipeline script (Windows-compatible)
│   └── requirements.txt
│
├── pyannote-whisper-diarization/
│   ├── ground_truth/                  # Reference JSONs (Teacher/Student turns)
│   ├── results/                       # Pipeline outputs (gitignored except der_results.csv)
│   ├── run_pipeline.py                # Main pipeline script
│   └── der.py                         # Sequence match evaluation
│
├── preprocessing/
│   ├── prepare_data.ipynb             # Parse PDFs → RTTMs + JSON ground truth
│   └── extract_transcripts.ipynb      # Extract text from transcripts
│
├── transcripts/                       # Source PDF transcripts (141 sessions)
└── literature/                        # Related papers
```

---

## Setup

### NeMo Pipeline

**Requirements:** NVIDIA GPU, CUDA, conda (Python 3.10)

```bash
cd nemo-multistage-classroom-diarization

# Install dependencies
pip install -r requirements.txt

# Fix torch for CUDA (CPU-only is installed by default)
pip install torch==2.6.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall --no-deps

# Install ffmpeg (needed by Whisper)
conda install -c conda-forge ffmpeg -y

# Clone NeMo (required for the diarizer)
git clone https://github.com/NVIDIA/NeMo.git

# Download wav2vec2-robust-large checkpoint
mkdir -p checkpoints/w2v2-robust-large-ckpt
gdown --fuzzy 'https://drive.google.com/file/d/1f9mMqzpGaLA2RB0m7dcesxo4deOB_GDq/view?usp=sharing' \
  -O checkpoints/w2v2-robust-large-ckpt/ckpt.pt
```

**Prepare data** (generates manifest + ground truth RTTMs from PDF transcripts):
```bash
jupyter nbconvert --to notebook --execute preprocessing/prepare_data.ipynb
```

**Run the full pipeline:**
```bash
./run_my_data.sh
```

> **Windows note:** The script uses `python` (not `python3`), `num_workers=0`, and avoids `bc` for float comparisons. Two NeMo package-level patches are required — see the [NeMo README](nemo-multistage-classroom-diarization/README.md) for details.

---

### Pyannote Pipeline

**Requirements:** NVIDIA GPU, CUDA, conda (Python 3.10), [HuggingFace token](https://huggingface.co/pyannote/speaker-diarization-3.1) with access to `pyannote/speaker-diarization-3.1`

```bash
# Create a separate conda environment (avoids conflicts with NeMo)
conda create -n pyannote-diarization python=3.10 -y
conda activate pyannote-diarization

pip install pyannote.audio==4.0.4 openai-whisper torchaudio librosa tqdm
pip install torch==2.6.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall --no-deps

# Store your HuggingFace token (never commit this file)
echo "hf_YOUR_TOKEN_HERE" > pyannote-whisper-diarization/hf_token.txt
```

**Run the pipeline:**
```bash
cd pyannote-whisper-diarization
python run_pipeline.py
```

**Evaluate:**
```bash
python der.py
```

Results are saved to `results/der_results.csv`.

---

## Ground Truth

Ground truth is derived from manually annotated PDF transcripts in `transcripts/`. The `prepare_data.ipynb` notebook parses timestamps and speaker labels from these PDFs and outputs:

- **RTTMs** → `nemo-multistage-classroom-diarization/ground_truth/rttm/` (for NeMo DER)
- **JSON turn sequences** → `pyannote-whisper-diarization/ground_truth/` (for sequence match evaluation)

Speaker labels: `T` = Teacher, `S`/`SN`/`Ss`/`SS` = Student. All other labels are skipped.

---

## Evaluation

The NeMo pipeline is evaluated with standard **DER** (False Alarm + Missed Speech + Confusion Error Rate).

The pyannote pipeline is evaluated with a **sequence match %** metric that compares the Teacher/Student turn alternation sequence (e.g. `[T, S, T, T, S, ...]`) between hypothesis and reference using `difflib.SequenceMatcher`. Role-flip detection is applied: if swapping Teacher↔Student yields a better score, the flipped version is used.

---

## Citation

If you use the NeMo pipeline, please cite the original paper:

```bibtex
@inproceedings{khan2025a,
  title     = {Multi-Stage Speaker Diarization for Noisy Classrooms},
  author    = {Khan, Ali Sartaz and Ogunremi, Tolulope and Attia, Ahmed Adel and Demszky, Dorottya},
  booktitle = {Proceedings of the 18th International Conference on Educational Data Mining},
  year      = {2025},
  doi       = {10.5281/zenodo.15870278}
}
```
