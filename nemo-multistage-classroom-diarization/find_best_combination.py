def parse_line(line):
    parts = line.strip().split('|')
    hyperparams = parts[0].strip().split(',')
    metrics = parts[1:]

    alpha = float(hyperparams[0].split('=')[1])
    onset = float(hyperparams[1].split('=')[1])
    offset = float(hyperparams[2].split('=')[1])
    der = float(metrics[3].split(':')[1].strip())

    return {'alpha': alpha, 'onset': onset, 'offset': offset, 'DER': der, 'line': line.strip()}

def find_best(lines, alpha_condition):
    best = None
    for line in lines:
        if not line.strip():
            continue
        parsed = parse_line(line)
        if alpha_condition(parsed['alpha']):
            if best is None or parsed['DER'] < best['DER']:
                best = parsed
    return best

with open('DER_results_my_data.txt', 'r') as f:
    lines = f.readlines()

best_eq_1 = find_best(lines, lambda a: a == 1)
best_neq_1 = find_best(lines, lambda a: a != 1)
best_overall = find_best(lines, lambda a: True)

print("Best overall:")
print(" ", best_overall['line'] if best_overall else "None")

print("\nBest with alpha == 1 (Whisper-only VAD):")
print(" ", best_eq_1['line'] if best_eq_1 else "None")

print("\nBest with alpha != 1 (combined VAD):")
print(" ", best_neq_1['line'] if best_neq_1 else "None")
