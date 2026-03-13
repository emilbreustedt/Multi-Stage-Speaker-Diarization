import json
import os
import difflib

def get_speaker_turns(data):
    """
    Collapses consecutive utterances by the same speaker into a single 'turn'.
    Example: [T, T, S, T] becomes [T, S, T]
    """
    turns = []
    for entry in data:
        # Standardize speaker keys
        speaker = entry.get("final_role", entry.get("speaker", "UNKNOWN"))
        # Only add if the speaker is different from the last one (Ping-pong logic)
        if not turns or turns[-1] != speaker:
            turns.append(speaker)
    return turns

def calculate_sequence_score(ref_turns, hyp_turns):
    """
    Uses SequenceMatcher to calculate alignment accuracy.
    """
    matcher = difflib.SequenceMatcher(None, ref_turns, hyp_turns)
    return matcher.ratio() * 100

def compare_folders_sequence(ref_folder, hyp_folder):
    files = [f for f in os.listdir(ref_folder) if f.endswith('.json')]
    
    all_scores = []
    
    print(f"{'Filename':<30} | {'Seq Match %':<12} | {'Turns (Ref/Hyp)':<15}")
    print("-" * 70)

    for filename in files:
        ref_path = os.path.join(ref_folder, filename)
        hyp_path = os.path.join(hyp_folder, filename)
        
        if not os.path.exists(hyp_path):
            continue
            
        try:
            with open(ref_path, 'r', encoding='utf-8') as r, open(hyp_path, 'r', encoding='utf-8') as h:
                # Clean invalid backslashes (common in AI math transcripts)
                ref_data = json.loads(r.read().replace('\\', '\\\\'))
                hyp_data = json.loads(h.read().replace('\\', '\\\\'))
                
                ref_turns = get_speaker_turns(ref_data)
                hyp_turns = get_speaker_turns(hyp_data)
                
                score = calculate_sequence_score(ref_turns, hyp_turns)
                all_scores.append(score)
                
                turn_counts = f"{len(ref_turns)}/{len(hyp_turns)}"
                print(f"{filename:<30} | {score:>10.2f}% | {turn_counts:>15}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Calculate and print the global average
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        print("-" * 70)
        print(f"DATASET AVERAGE SEQUENCE MATCH: {avg_score:.2f}%")
    else:
        print("No matching files found for comparison.")

# --- PATHS ---
REFERENCE_DIR = "combined_results"
HYPOTHESIS_DIR = "last_one"

if __name__ == "__main__":
    compare_folders_sequence(REFERENCE_DIR, HYPOTHESIS_DIR)
