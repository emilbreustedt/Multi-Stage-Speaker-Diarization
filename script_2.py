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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import json
from collections import defaultdict

File_name = "" # Global variable to hold filename
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
    HF_TOKEN = "YOUR_HF_TOKEN"
    
    # Paths
    DATA_DIR = "classroom_recordings"
    ANNOTATIONS_FILE = "annotations.csv"
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
            self.config.DIARIZATION_MODEL
        )
        
        if torch.cuda.is_available():
            self.diarization_pipeline.to(torch.device("cuda"))
        
        print(f"Loading Whisper model ({self.config.WHISPER_MODEL})...")
        self.whisper_model = whisper.load_model(self.config.WHISPER_MODEL).to("cuda")
    
    def process_audio(self, audio_path):
        """
        Process audio file with diarization and transcription.
        Returns aligned results with speaker labels and transcripts.
        """
        if self.diarization_pipeline is None or self.whisper_model is None:
            self.load_models()
        
        print(f"\nProcessing: {audio_path}")
        
        # Step 1: Perform diarization
        print("  → Running speaker diarization...")
        diarization = self.diarization_pipeline(audio_path)
        
        # Step 2: Get Whisper transcription with timestamps
        print("  → Running Whisper transcription...")
        whisper_result = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=True,
            language='de',  # or None for auto-detect
            fp16=torch.cuda.is_available()
        )
        
        # Step 3: Align diarization with transcription
        print("  → Aligning speakers with transcripts...")
        aligned_results = self.align_diarization_transcription(
            diarization, whisper_result
        )        
        return aligned_results
    
    def align_diarization_transcription(self, diarization, whisper_result):
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
            # Use the middle of the segment for speaker assignment
            mid_time = (start_time + end_time) / 2
            
            speaker = None
            annotation = diarization.speaker_diarization
            
            for turn, _, spk in annotation.itertracks(yield_label=True):
                if turn.start <= mid_time <= turn.end:
                    speaker = spk
                    print("from main: ", speaker)
                    break
            
            # If no exact match, find the speaker with most overlap
            if speaker is None:
                speaker = self._find_overlapping_speaker(
                    diarization, start_time, end_time
                )
                print("overlapped: ", speaker)
            
            aligned_segments.append({
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'speaker': speaker if speaker else 'UNKNOWN',
                'text': text
            })

        with open("debug_diarization.json", "w") as f:
            json.dump(aligned_segments, f, indent=4)
        
        return aligned_segments
    
    def _find_overlapping_speaker(self, diarization, start, end):
        """Find speaker with maximum overlap with time range"""
        max_overlap = 0
        best_speaker = None
        annotation = diarization.speaker_diarization
        
        for turn, _, speaker in annotation.itertracks(yield_label=True):
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
            'long_segments': 0,  # segments > 10s
            'questions_asked': 0,  # heuristic from text
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
        # Step 1: Collect features for each speaker
        self._collect_speaker_features(aligned_segments)
        
        # Step 2: Calculate discriminative features
        features_df = self._calculate_discriminative_features()
        
        # Step 3: Score-based identification
        teacher_scores = self._score_speakers(features_df)
        
        # Step 4: Identify teacher (highest score)
        identified_roles = self._assign_roles(teacher_scores)
        
        return identified_roles, teacher_scores
    
    def _collect_speaker_features(self, segments):
        """Collect basic statistics for each speaker"""
        
        for i, seg in enumerate(segments):
            speaker = seg['speaker']
            
            # Duration and word count
            self.speaker_features[speaker]['total_duration'] += seg['duration']
            self.speaker_features[speaker]['num_segments'] += 1
            self.speaker_features[speaker]['total_words'] += len(seg['text'].split())
            self.speaker_features[speaker]['segments'].append(seg)
            
            # Long segment detection (teachers often have longer explanations)
            if seg['duration'] > 10:
                self.speaker_features[speaker]['long_segments'] += 1
            
            # Question detection (teachers ask more questions)
            if '?' in seg['text']:
                self.speaker_features[speaker]['questions_asked'] += 1
            
            # Speaking turn tracking
            self.speaker_features[speaker]['speaking_turns'].append({
                'start': seg['start'],
                'end': seg['end']
            })
    
    def _calculate_discriminative_features(self):
        """Calculate features that discriminate teacher from students"""
        
        features = {}
        
        for speaker, stats in self.speaker_features.items():
            
            # Average segment duration
            if stats['num_segments'] > 0:
                avg_duration = stats['total_duration'] / stats['num_segments']
            else:
                avg_duration = 0
            
            # Words per minute
            if stats['total_duration'] > 0:
                wpm = (stats['total_words'] / stats['total_duration']) * 60
            else:
                wpm = 0
            
            # Proportion of long segments
            if stats['num_segments'] > 0:
                long_seg_ratio = stats['long_segments'] / stats['num_segments']
            else:
                long_seg_ratio = 0
            
            # Questions per minute
            if stats['total_duration'] > 0:
                qpm = (stats['questions_asked'] / stats['total_duration']) * 60
            else:
                qpm = 0
            
            # Calculate speaking continuity (fewer, longer turns = teacher)
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
        
        # Continuity score: lower gaps = higher continuity
        # Teachers typically have more continuous speaking
        continuity_score = 1.0 / (1.0 + avg_gap)
        
        return {'avg_gap': avg_gap, 'continuity_score': continuity_score}
    
    def _score_speakers(self, features):
        """
        Score each speaker on likelihood of being a teacher.
        
        Teacher indicators:
        - Longer total speaking time
        - Longer average segment duration
        - More total words
        - Higher ratio of long segments
        - More questions asked
        """
        
        scores = {}
        
        if not features:
            return scores
        
        # Normalize features across all speakers
        all_speakers = list(features.keys())
        
        # Get max values for normalization
        max_duration = max(f['total_duration'] for f in features.values())
        max_words = max(f['total_words'] for f in features.values())
        max_avg_duration = max(f['avg_segment_duration'] for f in features.values())
        
        for speaker in all_speakers:
            f = features[speaker]
            
            # Weighted scoring (using config weights)
            score = 0.0
            
            # Duration-based
            if max_duration > 0:
                score += self.config.WEIGHT_DURATION * (f['total_duration'] / max_duration)
            
            # Word count
            if max_words > 0:
                score += self.config.WEIGHT_WORDS * (f['total_words'] / max_words)
            
            # Average segment duration
            if max_avg_duration > 0:
                score += self.config.WEIGHT_AVG_DURATION * (f['avg_segment_duration'] / max_avg_duration)
            
            # Long segment ratio
            score += self.config.WEIGHT_LONG_SEGMENTS * f['long_segment_ratio']
            
            # Questions
            score += self.config.WEIGHT_QUESTIONS * min(f['questions_per_minute'] / 5.0, 1.0)
            
            # Turn continuity
            score += self.config.WEIGHT_CONTINUITY * f['turn_continuity']
            
            scores[speaker] = {
                'score': score,
                'features': f
            }
        
        return scores
    
    def _assign_roles(self, teacher_scores):
        """
        Assign teacher/student roles based on scores.
        
        Strategy: Highest score = teacher, rest = students
        """
        
        if not teacher_scores:
            return {}
        
        # Sort by score
        sorted_speakers = sorted(
            teacher_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        roles = {}
        
        # Top scorer is teacher
        teacher_speaker = sorted_speakers[0][0]
        roles[teacher_speaker] = 'Teacher'
        
        # Rest are students
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
    
    Returns a rich dataset with speaker roles and transcripts.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Step 1: Diarization + Transcription
    print("="*70)
    print("STEP 1: DIARIZATION + TRANSCRIPTION")
    print("="*70)
    processor_obj = ClassroomProcessor(config)
    aligned_segments = processor_obj.process_audio(audio_path)
    
    print(f"\n✓ Found {len(aligned_segments)} transcribed segments")
    
    # Step 2: Enhanced Teacher Identification (Heuristic-based)
    print("\n" + "="*70)
    print("STEP 2: ENHANCED TEACHER IDENTIFICATION")
    print("="*70)
    
    identifier = EnhancedTeacherIdentifier(config)
    identified_roles, teacher_scores = identifier.analyze_speakers(aligned_segments)
    
    # Print analysis
    identifier.print_analysis(teacher_scores, identified_roles)
    
    # Step 3: Optional Classifier Enhancement
    if use_classifier and classifier_model_path and os.path.exists(classifier_model_path):
        print("\n" + "="*70)
        print("STEP 3: ACOUSTIC CLASSIFIER REFINEMENT (Optional)")
        print("="*70)
        
        wav2vec_processor = Wav2Vec2Processor.from_pretrained(config.WAV2VEC_MODEL)
        classifier = TeacherStudentClassifier(config).to(device)
        classifier.load_state_dict(torch.load(classifier_model_path, map_location=device))
        classifier.eval()
        
        # Classify each segment and combine with heuristic
        print("Classifying speaker roles...")
        for segment in tqdm(aligned_segments):
            if segment['duration'] < config.MIN_SEGMENT_DURATION:
                segment['classifier_prediction'] = 'UNKNOWN'
                segment['classifier_confidence'] = 0.0
                continue
            
            # Extract audio
            audio_segment = processor_obj.extract_segment_audio(
                audio_path, segment['start'], segment['end']
            )
            
            # Truncate if too long
            max_len = int(config.MAX_SEGMENT_DURATION * config.SAMPLE_RATE)
            if len(audio_segment) > max_len:
                audio_segment = audio_segment[:max_len]
            
            # Classify
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
    
    # Step 4: Add final roles to segments
    for segment in aligned_segments:
        segment['final_role'] = identified_roles.get(segment['speaker'], 'UNKNOWN')
    
    # Step 5: Calculate speaker statistics
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
# Output and Visualization
# =============================================================================
def save_results_to_file(segments, output_path):
    """Save results to CSV and text files"""
    
    # Save as CSV
    df = pd.DataFrame(segments)
    csv_path = output_path.replace('.txt', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results to {csv_path}")
    
    # Save as readable transcript
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("CLASSROOM TRANSCRIPT\n")
        f.write("="*70 + "\n\n")
        
        current_speaker = None
        for seg in segments:
            role = seg.get('final_role', 'UNKNOWN')
            
            # Add speaker label when speaker changes
            if role != current_speaker:
                current_speaker = role
                speaker_id = seg.get('speaker', 'UNKNOWN')
                f.write(f"\n[{current_speaker} - {speaker_id}]\n")
            
            f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}\n")
    file_namee = File_name.split("/")[-1]
    print("file_namee: ", file_namee)
    folder_name = file_namee.split("_")[0]
    os.makedirs(folder_name, exist_ok=True)

    file_name = "/home/bethge/bkr426/diarization/diarized_whisper_v3_trubo_german" +"/"+folder_name+ "/" + file_namee.split("_")[1].split(".")[0]

    with open(f"{file_name}.json", 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=4)
        
    
    print(f"✓ Saved transcript to {output_path}")

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
    
    # Show sample transcript
    print("\n" + "-"*70)
    print("Sample Transcript (first 5 segments):")
    print("-"*70)
    
    for seg in segments[:5]:
        print(f"\n[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['final_role']}")
        print(f"  \"{seg['text']}\"")

# =============================================================================
# Training Dataset Creation from Diarization + Transcription
# =============================================================================
def create_training_data_from_recordings(audio_files, output_csv, config):
    """
    Process multiple classroom recordings with diarization + transcription
    to create training dataset. You'll need to manually label the roles.
    """
    processor = ClassroomProcessor(config)
    
    all_segments = []
    
    for audio_path in audio_files:
        print(f"\nProcessing {audio_path}...")
        segments = processor.process_audio(audio_path)
        
        # Add filename
        for seg in segments:
            seg['filename'] = os.path.basename(audio_path)
        
        all_segments.extend(segments)
    
    # Save to CSV for manual annotation
    df = pd.DataFrame(all_segments)
    df['label'] = ''  # Empty - to be filled manually
    df['teacher_id'] = ''  # Empty - to be filled manually
    
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved {len(all_segments)} segments to {output_csv}")
    print("→ Please manually fill 'label' (teacher/student) and 'teacher_id' columns")

# =============================================================================
# Usage Examples
# =============================================================================
def main():
    config = Config()
    
    # =========================================================================
    # Example 1: Create training data from recordings (Optional)
    # =========================================================================
    # classroom_files = [
    #     "classroom_recording_1.wav",
    #     "classroom_recording_2.wav",
    #     "classroom_recording_3.wav",
    # ]
    # create_training_data_from_recordings(
    #     classroom_files,
    #     "training_segments.csv",
    #     config
    # )
    
    # =========================================================================
    # Example 2: Full analysis with enhanced teacher identification
    # =========================================================================
    segments, speaker_stats, identified_roles, teacher_scores = full_classroom_analysis(
        audio_path=File_name,
        config=config,
        use_classifier=False,  # Set to True if you have a trained model
        classifier_model_path=config.MODEL_SAVE_PATH
    )
    
    # Print summary
    print_summary(segments, speaker_stats, identified_roles, teacher_scores)
    
    # Save results
    save_results_to_file(segments, "classroom_transcript.txt")
    
    # =========================================================================
    # Example 3: Detailed analysis and statistics
    # =========================================================================
    # Create detailed DataFrame
    df = pd.DataFrame(segments)
    
    # Filter teacher utterances
    teacher_df = df[df['final_role'] == 'Teacher']
    print(f"\n" + "="*70)
    print("TEACHER STATISTICS")
    print("="*70)
    print(f"Teacher spoke {len(teacher_df)} times")
    print(f"Teacher total speaking time: {teacher_df['duration'].sum():.1f}s")
    print(f"Teacher total words: {teacher_df['text'].apply(lambda x: len(x.split())).sum()}")
    
    # Filter student utterances
    student_df = df[df['final_role'] == 'Student']
    print(f"\n" + "="*70)
    print("STUDENT STATISTICS")
    print("="*70)
    print(f"Students spoke {len(student_df)} times")
    print(f"Students total speaking time: {student_df['duration'].sum():.1f}s")
    print(f"Students total words: {student_df['text'].apply(lambda x: len(x.split())).sum()}")
    
    # Calculate talk ratio
    teacher_time = teacher_df['duration'].sum()
    student_time = student_df['duration'].sum()
    total_time = teacher_time + student_time
    
    if total_time > 0:
        teacher_pct = (teacher_time / total_time) * 100
        student_pct = (student_time / total_time) * 100
        
        print(f"\n" + "="*70)
        print("TALK TIME DISTRIBUTION")
        print("="*70)
        print(f"Teacher: {teacher_time:.0f}s ({teacher_pct:.1f}%)")
        print(f"Students: {student_time:.0f}s ({student_pct:.1f}%)")
        print(f"Ratio → Teacher:Student = {teacher_time:.0f}:{student_time:.0f}")
    
    # Save speaker-level summary
    speaker_summary = []
    for speaker, role in identified_roles.items():
        stats = speaker_stats[speaker]
        score_data = teacher_scores.get(speaker, {})
        
        speaker_summary.append({
            'speaker_id': speaker,
            'role': role,
            'score': score_data.get('score', 0.0) if score_data else 0.0,
            'total_duration_s': stats['total_duration'],
            'num_segments': stats['num_segments'],
            'total_words': stats['total_words'],
            'avg_segment_duration_s': stats['total_duration'] / stats['num_segments'] if stats['num_segments'] > 0 else 0
        })
    
    summary_df = pd.DataFrame(speaker_summary)
    summary_df.to_csv("speaker_summary.csv", index=False)
    print(f"\n✓ Saved speaker summary to speaker_summary.csv")