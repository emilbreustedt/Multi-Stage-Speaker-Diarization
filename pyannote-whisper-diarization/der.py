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
    Also tests role-flipped hypothesis (Teacher↔Student) and returns the better score.
    """
    FLIP = {"Teacher": "Student", "Student": "Teacher"}
    hyp_flipped = [FLIP.get(t, t) for t in hyp_turns]
    score_normal = difflib.SequenceMatcher(None, ref_turns, hyp_turns).ratio() * 100
    score_flipped = difflib.SequenceMatcher(None, ref_turns, hyp_flipped).ratio() * 100
    return max(score_normal, score_flipped), score_flipped > score_normal

def compare_folders_sequence(ref_folder, hyp_folder, output_csv="der_results.csv"):
    files = sorted(f for f in os.listdir(ref_folder) if f.endswith('.json'))

    all_scores = []
    rows = []

    print(f"{'Filename':<30} | {'Seq Match %':<12} | {'Turns (Ref/Hyp)':<15} | {'Note'}")
    print("-" * 80)

    for filename in files:
        ref_path = os.path.join(ref_folder, filename)
        hyp_path = os.path.join(hyp_folder, filename)

        if not os.path.exists(hyp_path):
            continue

        try:
            with open(ref_path, 'r', encoding='utf-8') as r, open(hyp_path, 'r', encoding='utf-8') as h:
                ref_data = json.loads(r.read().replace('\\', '\\\\'))
                hyp_data = json.loads(h.read().replace('\\', '\\\\'))

                ref_turns = get_speaker_turns(ref_data)
                hyp_turns = get_speaker_turns(hyp_data)

                score, was_flipped = calculate_sequence_score(ref_turns, hyp_turns)
                all_scores.append(score)

                turn_counts = f"{len(ref_turns)}/{len(hyp_turns)}"
                note = "ROLE FLIPPED" if was_flipped else ""
                print(f"{filename:<30} | {score:>10.2f}% | {turn_counts:>15} | {note}")
                rows.append({"filename": filename.replace('.json', ''),
                             "seq_match_pct": round(score, 2),
                             "turns_ref": len(ref_turns),
                             "turns_hyp": len(hyp_turns),
                             "role_flipped": was_flipped})

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Calculate and print the global average
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        print("-" * 70)
        print(f"DATASET AVERAGE SEQUENCE MATCH: {avg_score:.2f}%")
        rows.append({"filename": "AVERAGE", "seq_match_pct": round(avg_score, 2),
                     "turns_ref": "", "turns_hyp": "", "role_flipped": ""})
    else:
        print("No matching files found for comparison.")

    # Save to CSV
    if rows:
        header = "filename;seq_match_pct;turns_ref;turns_hyp;role_flipped\n"
        lines = [f"{r['filename']};{r['seq_match_pct']};{r['turns_ref']};{r['turns_hyp']};{r['role_flipped']}" for r in rows]
        with open(output_csv, 'w', encoding='utf-8') as f:
            f.write(header + "\n".join(lines) + "\n")
        print(f"Results saved to {output_csv}")

# --- PATHS (relative to this script's location) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(_HERE, "ground_truth")
HYPOTHESIS_DIR = os.path.join(_HERE, "results")
OUTPUT_CSV    = os.path.join(_HERE, "results", "der_results.csv")

if __name__ == "__main__":
    compare_folders_sequence(REFERENCE_DIR, HYPOTHESIS_DIR, OUTPUT_CSV)
