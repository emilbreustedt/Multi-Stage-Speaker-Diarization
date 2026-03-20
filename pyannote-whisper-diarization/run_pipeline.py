import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import whisper
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from pyannote.audio import Pipeline
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import json
from collections import defaultdict

# =============================================================================
# Configuration
# =============================================================================
class Config:
    # Audio parameters
    SAMPLE_RATE = 16000
    MIN_SEGMENT_DURATION = 1.0
    MAX_SEGMENT_DURATION = 5.0

    # Model parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    HIDDEN_DIM = 128

    # Wav2Vec2 parameters
    WAV2VEC_MODEL = "facebook/wav2vec2-base"
    FREEZE_WAV2VEC = True

    # Whisper parameters
    WHISPER_MODEL = "large-v2"  # tiny, base, small, medium, large

    # Diarization
    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
    HF_TOKEN = open(Path(__file__).parent / "hf_token.txt").read().strip()

    # Paths
    AUDIO_DIR = r"D:\audio"
    RESULTS_DIR = "results"           # relative to this script's location
    MODEL_SAVE_PATH = "teacher_student_classifier.pth"

    # Teacher identification weights (adjust based on your classroom type)
    WEIGHT_DURATION = 0.30      # Total speaking time
    WEIGHT_WORDS = 0.20         # Total words spoken
    WEIGHT_AVG_DURATION = 0.20  # Average segment length
    WEIGHT_LONG_SEGMENTS = 0.15 # Proportion of long explanations
    WEIGHT_QUESTIONS = 0.10     # Questions asked
    WEIGHT_CONTINUITY = 0.05    # Speaking continuity

# =============================================================================
# Combined Diarization + Transcription
# =============================================================================
class ClassroomProcessor:
    """
    Combines speaker diarization with Whisper transcription.
    """

    def __init__(self, config):
        self.config = config
        self.diarization_pipeline = None
        self.whisper_model = None

    def load_models(self):
        """Load diarization and transcription models"""
        print("Loading pyannote diarization pipeline...")

        self.diarization_pipeline = Pipeline.from_pretrained(
            self.config.DIARIZATION_MODEL,
            token=self.config.HF_TOKEN
        )

        if torch.cuda.is_available():
            self.diarization_pipeline.to(torch.device("cuda"))

        print(f"Loading Whisper model ({self.config.WHISPER_MODEL})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model(self.config.WHISPER_MODEL).to(device)

    def process_audio(self, audio_path):
        """
        Process audio file with diarization and transcription.
        Returns aligned results with speaker labels and transcripts.
        """
        if self.diarization_pipeline is None or self.whisper_model is None:
            self.load_models()

        print(f"\nProcessing: {audio_path}")

        # Step 1: Perform diarization (pass stereo as mono via waveform dict)
        print("  → Running speaker diarization...")
        import torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:  # stereo -> mono
            waveform = waveform.mean(dim=0, keepdim=True)
        diarization_output = self.diarization_pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            min_speakers=2,
            max_speakers=10,
        )
        # pyannote 4.x returns DiarizeOutput — extract the Annotation
        diarization = diarization_output.speaker_diarization

        # Step 2: Get Whisper transcription with timestamps
        print("  → Running Whisper transcription...")
        whisper_result = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=True,
            language='de',
            fp16=torch.cuda.is_available()
        )

        # Step 3: Align diarization with transcription
        print("  → Aligning speakers with transcripts...")
        aligned_results = self.align_diarization_transcription(
            diarization, whisper_result, audio_path
        )
        return aligned_results

    def align_diarization_transcription(self, diarization, whisper_result, audio_path):
        """
        Align speaker diarization with Whisper word-level transcription.
        """
        aligned_segments = []

        # Process each segment from Whisper
        for segment in whisper_result['segments']:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()

            if not text:
                continue

            # Find which speaker was active during this segment
            mid_time = (start_time + end_time) / 2

            speaker = None
            # pyannote pipeline returns an Annotation object directly
            for turn, _, spk in diarization.itertracks(yield_label=True):
                if turn.start <= mid_time <= turn.end:
                    speaker = spk
                    break

            # If no exact match, find the speaker with most overlap
            if speaker is None:
                speaker = self._find_overlapping_speaker(
                    diarization, start_time, end_time
                )

            aligned_segments.append({
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'speaker': speaker if speaker else 'UNKNOWN',
                'text': text
            })

        # Save debug JSON to results folder
        results_dir = Path(self.config.RESULTS_DIR)
        results_dir.mkdir(exist_ok=True)
        stem = Path(audio_path).stem
        debug_path = results_dir / f"{stem}_debug.json"
        with open(debug_path, "w", encoding='utf-8') as f:
            json.dump(aligned_segments, f, indent=4)

        return aligned_segments

    def _find_overlapping_speaker(self, diarization, start, end):
        """Find speaker with maximum overlap with time range"""
        max_overlap = 0
        best_speaker = None

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(turn.start, start)
            overlap_end = min(turn.end, end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker

        return best_speaker

    def extract_segment_audio(self, audio_path, start, end):
        """Extract audio segment"""
        audio, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        return audio[start_sample:end_sample]

# =============================================================================
# Enhanced Teacher-Student Identification
# =============================================================================
class EnhancedTeacherIdentifier:
    """
    Multi-signal approach to identify teacher vs students.
    Uses: speaking patterns, duration, prosody, and text analysis.
    """

    def __init__(self, config):
        self.config = config
        self.speaker_features = defaultdict(lambda: {
            'total_duration': 0,
            'num_segments': 0,
            'total_words': 0,
            'avg_segment_duration': 0,
            'long_segments': 0,
            'questions_asked': 0,
            'interruptions': 0,
            'speaking_turns': [],
            'segments': []
        })

    def analyze_speakers(self, aligned_segments):
        """
        Analyze all speakers and identify teacher using multiple heuristics.

        Returns:
            dict: speaker -> role mapping
            dict: detailed scores for each speaker
        """
        self._collect_speaker_features(aligned_segments)
        features_df = self._calculate_discriminative_features()
        teacher_scores = self._score_speakers(features_df)
        identified_roles = self._assign_roles(teacher_scores)
        return identified_roles, teacher_scores

    def _collect_speaker_features(self, segments):
        """Collect basic statistics for each speaker"""
        for i, seg in enumerate(segments):
            speaker = seg['speaker']

            self.speaker_features[speaker]['total_duration'] += seg['duration']
            self.speaker_features[speaker]['num_segments'] += 1
            self.speaker_features[speaker]['total_words'] += len(seg['text'].split())
            self.speaker_features[speaker]['segments'].append(seg)

            if seg['duration'] > 10:
                self.speaker_features[speaker]['long_segments'] += 1

            if '?' in seg['text']:
                self.speaker_features[speaker]['questions_asked'] += 1

            self.speaker_features[speaker]['speaking_turns'].append({
                'start': seg['start'],
                'end': seg['end']
            })

    def _calculate_discriminative_features(self):
        """Calculate features that discriminate teacher from students"""
        features = {}

        for speaker, stats in self.speaker_features.items():

            avg_duration = (stats['total_duration'] / stats['num_segments']
                            if stats['num_segments'] > 0 else 0)

            wpm = ((stats['total_words'] / stats['total_duration']) * 60
                   if stats['total_duration'] > 0 else 0)

            long_seg_ratio = (stats['long_segments'] / stats['num_segments']
                              if stats['num_segments'] > 0 else 0)

            qpm = ((stats['questions_asked'] / stats['total_duration']) * 60
                   if stats['total_duration'] > 0 else 0)

            turn_gaps = self._calculate_turn_continuity(stats['speaking_turns'])

            features[speaker] = {
                'total_duration': stats['total_duration'],
                'num_segments': stats['num_segments'],
                'avg_segment_duration': avg_duration,
                'words_per_minute': wpm,
                'total_words': stats['total_words'],
                'long_segment_ratio': long_seg_ratio,
                'questions_per_minute': qpm,
                'avg_turn_gap': turn_gaps['avg_gap'],
                'turn_continuity': turn_gaps['continuity_score']
            }

        return features

    def _calculate_turn_continuity(self, turns):
        """Calculate how continuous speaking turns are"""
        if len(turns) < 2:
            return {'avg_gap': 0, 'continuity_score': 1.0}

        gaps = []
        for i in range(len(turns) - 1):
            gap = turns[i+1]['start'] - turns[i]['end']
            gaps.append(gap)

        avg_gap = np.mean(gaps) if gaps else 0
        continuity_score = 1.0 / (1.0 + avg_gap)

        return {'avg_gap': avg_gap, 'continuity_score': continuity_score}

    def _score_speakers(self, features):
        """Score each speaker on likelihood of being a teacher."""
        scores = {}

        if not features:
            return scores

        max_duration = max(f['total_duration'] for f in features.values())
        max_words = max(f['total_words'] for f in features.values())
        max_avg_duration = max(f['avg_segment_duration'] for f in features.values())

        for speaker, f in features.items():
            score = 0.0

            if max_duration > 0:
                score += self.config.WEIGHT_DURATION * (f['total_duration'] / max_duration)
            if max_words > 0:
                score += self.config.WEIGHT_WORDS * (f['total_words'] / max_words)
            if max_avg_duration > 0:
                score += self.config.WEIGHT_AVG_DURATION * (f['avg_segment_duration'] / max_avg_duration)

            score += self.config.WEIGHT_LONG_SEGMENTS * f['long_segment_ratio']
            score += self.config.WEIGHT_QUESTIONS * min(f['questions_per_minute'] / 5.0, 1.0)
            score += self.config.WEIGHT_CONTINUITY * f['turn_continuity']

            scores[speaker] = {
                'score': score,
                'features': f
            }

        return scores

    def _assign_roles(self, teacher_scores):
        """Highest score = teacher, rest = students."""
        if not teacher_scores:
            return {}

        sorted_speakers = sorted(
            teacher_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        roles = {}
        roles[sorted_speakers[0][0]] = 'Teacher'
        for speaker, _ in sorted_speakers[1:]:
            roles[speaker] = 'Student'

        return roles

    def print_analysis(self, teacher_scores, identified_roles):
        """Print detailed analysis"""
        print("\n" + "="*70)
        print("SPEAKER ROLE IDENTIFICATION ANALYSIS")
        print("="*70)

        sorted_speakers = sorted(
            teacher_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        for speaker, data in sorted_speakers:
            role = identified_roles.get(speaker, 'UNKNOWN')
            score = data['score']
            features = data['features']

            print(f"\n{speaker} → {role} (Score: {score:.3f})")
            print(f"  Total speaking time: {features['total_duration']:.1f}s")
            print(f"  Number of segments: {features['num_segments']}")
            print(f"  Average segment duration: {features['avg_segment_duration']:.2f}s")
            print(f"  Total words: {features['total_words']}")
            print(f"  Words per minute: {features['words_per_minute']:.1f}")
            print(f"  Long segments (>10s): {features['long_segment_ratio']:.1%}")
            print(f"  Questions per minute: {features['questions_per_minute']:.2f}")
            print(f"  Turn continuity: {features['turn_continuity']:.3f}")

# =============================================================================
# Teacher-Student Classifier (Optional - for future use with training data)
# =============================================================================
class TeacherStudentClassifier(nn.Module):
    def __init__(self, config):
        super(TeacherStudentClassifier, self).__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained(config.WAV2VEC_MODEL)

        if config.FREEZE_WAV2VEC:
            for param in self.wav2vec.parameters():
                param.requires_grad = False

        self.embedding_dim = self.wav2vec.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.Linear(config.HIDDEN_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, input_values):
        outputs = self.wav2vec(input_values)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(embeddings)
        return logits, embeddings

# =============================================================================
# Complete Pipeline: Diarization + Transcription + Enhanced Identification
# =============================================================================
def full_classroom_analysis(audio_path, config, use_classifier=False, classifier_model_path=None):
    """
    Complete pipeline combining:
    1. Diarization (who spoke when)
    2. Transcription (what was said)
    3. Enhanced Teacher Identification (heuristic-based)
    4. Optional: Acoustic classifier (if trained model available)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("STEP 1: DIARIZATION + TRANSCRIPTION")
    print("="*70)
    processor_obj = ClassroomProcessor(config)
    aligned_segments = processor_obj.process_audio(audio_path)

    print(f"\n✓ Found {len(aligned_segments)} transcribed segments")

    print("\n" + "="*70)
    print("STEP 2: ENHANCED TEACHER IDENTIFICATION")
    print("="*70)

    identifier = EnhancedTeacherIdentifier(config)
    identified_roles, teacher_scores = identifier.analyze_speakers(aligned_segments)
    identifier.print_analysis(teacher_scores, identified_roles)

    if use_classifier and classifier_model_path and os.path.exists(classifier_model_path):
        print("\n" + "="*70)
        print("STEP 3: ACOUSTIC CLASSIFIER REFINEMENT (Optional)")
        print("="*70)

        wav2vec_processor = Wav2Vec2Processor.from_pretrained(config.WAV2VEC_MODEL)
        classifier = TeacherStudentClassifier(config).to(device)
        classifier.load_state_dict(torch.load(classifier_model_path, map_location=device))
        classifier.eval()

        print("Classifying speaker roles...")
        for segment in tqdm(aligned_segments):
            if segment['duration'] < config.MIN_SEGMENT_DURATION:
                segment['classifier_prediction'] = 'UNKNOWN'
                segment['classifier_confidence'] = 0.0
                continue

            audio_segment = processor_obj.extract_segment_audio(
                audio_path, segment['start'], segment['end']
            )

            max_len = int(config.MAX_SEGMENT_DURATION * config.SAMPLE_RATE)
            if len(audio_segment) > max_len:
                audio_segment = audio_segment[:max_len]

            inputs = wav2vec_processor(
                audio_segment,
                sampling_rate=config.SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                logits, _ = classifier(inputs.input_values.to(device))
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1).item()

            segment['classifier_prediction'] = 'Teacher' if pred == 1 else 'Student'
            segment['classifier_confidence'] = probs[0][pred].item()

    for segment in aligned_segments:
        segment['final_role'] = identified_roles.get(segment['speaker'], 'UNKNOWN')

    speaker_stats = aggregate_speaker_stats(aligned_segments)

    return aligned_segments, speaker_stats, identified_roles, teacher_scores


def aggregate_speaker_stats(segments):
    """Aggregate statistics by speaker"""
    stats = defaultdict(lambda: {
        'total_duration': 0,
        'total_words': 0,
        'num_segments': 0,
        'segments': [],
        'role': 'UNKNOWN'
    })

    for seg in segments:
        speaker = seg['speaker']
        stats[speaker]['segments'].append(seg)
        stats[speaker]['total_duration'] += seg['duration']
        stats[speaker]['total_words'] += len(seg['text'].split())
        stats[speaker]['num_segments'] += 1
        stats[speaker]['role'] = seg.get('final_role', 'UNKNOWN')

    return dict(stats)

# =============================================================================
# Output
# =============================================================================
def save_results_to_file(segments, audio_path, config):
    """Save results to JSON and readable transcript in the results folder."""
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)

    stem = Path(audio_path).stem  # e.g. "T-1101-L-Lek1"

    # Save full segments as JSON
    json_path = results_dir / f"{stem}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=4)
    print(f"✓ Saved JSON to {json_path}")

    # Save as readable transcript
    txt_path = results_dir / f"{stem}_transcript.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("CLASSROOM TRANSCRIPT\n")
        f.write("="*70 + "\n\n")

        current_speaker = None
        for seg in segments:
            role = seg.get('final_role', 'UNKNOWN')
            if role != current_speaker:
                current_speaker = role
                speaker_id = seg.get('speaker', 'UNKNOWN')
                f.write(f"\n[{current_speaker} - {speaker_id}]\n")
            f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}\n")

    print(f"✓ Saved transcript to {txt_path}")


def print_summary(segments, speaker_stats, identified_roles, teacher_scores):
    """Print summary of analysis"""
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)

    print(f"\nTotal segments: {len(segments)}")
    print(f"Total speakers detected: {len(speaker_stats)}")

    print("\n" + "-"*70)
    print("Speaker Analysis:")
    print("-"*70)

    for speaker, role in identified_roles.items():
        stats = speaker_stats[speaker]
        score_data = teacher_scores.get(speaker, {})
        score = score_data.get('score', 0.0) if score_data else 0.0

        print(f"\n{speaker} → {role} (Score: {score:.3f})")
        print(f"  Speaking time: {stats['total_duration']:.1f}s")
        print(f"  Total words: {stats['total_words']}")
        print(f"  Segments: {stats['num_segments']}")

    print("\n" + "-"*70)
    print("Sample Transcript (first 5 segments):")
    print("-"*70)

    for seg in segments[:5]:
        print(f"\n[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['final_role']}")
        print(f"  \"{seg['text']}\"")

# =============================================================================
# Main — loop over all WAV files in AUDIO_DIR
# =============================================================================
def main():
    config = Config()
    audio_dir = Path(config.AUDIO_DIR)
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {audio_dir}")
        return

    print(f"Found {len(wav_files)} WAV files in {audio_dir}")

    # Load models once, reuse across files
    processor = ClassroomProcessor(config)
    processor.load_models()

    all_speaker_summaries = []

    for audio_path in wav_files:
        print(f"\n{'='*70}")
        print(f"Processing: {audio_path.name}")
        print(f"{'='*70}")

        try:
            segments, speaker_stats, identified_roles, teacher_scores = full_classroom_analysis(
                audio_path=str(audio_path),
                config=config,
                use_classifier=False,
                classifier_model_path=config.MODEL_SAVE_PATH
            )

            print_summary(segments, speaker_stats, identified_roles, teacher_scores)
            save_results_to_file(segments, audio_path, config)

            # Collect per-file speaker summary row
            for speaker, role in identified_roles.items():
                stats = speaker_stats[speaker]
                score_data = teacher_scores.get(speaker, {})
                all_speaker_summaries.append({
                    'file': audio_path.name,
                    'speaker_id': speaker,
                    'role': role,
                    'score': score_data.get('score', 0.0) if score_data else 0.0,
                    'total_duration_s': stats['total_duration'],
                    'num_segments': stats['num_segments'],
                    'total_words': stats['total_words'],
                    'avg_segment_duration_s': (stats['total_duration'] / stats['num_segments']
                                               if stats['num_segments'] > 0 else 0)
                })

        except Exception as e:
            print(f"ERROR processing {audio_path.name}: {e}")
            continue

    # Save combined speaker summary
    if all_speaker_summaries:
        summary_path = results_dir / "speaker_summary.csv"
        pd.DataFrame(all_speaker_summaries).to_csv(summary_path, index=False)
        print(f"\n✓ Saved speaker summary to {summary_path}")


if __name__ == "__main__":
    main()
