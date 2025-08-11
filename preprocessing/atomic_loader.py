# preprocessing/atomic_loader.py
import os
import csv

def load_atomic_tsv(filepath, max_triples=None):
    """
    Load ATOMIC CSV/TSV file (comma or tab separated).
    Returns list of (head, relation, tail) tuples in lowercase.
    """
    if not os.path.exists(filepath):
        print(f"[ATOMIC] Path not found: {filepath}")
        return []

    triples = []
    count = 0

    # Detect delimiter: tab for .tsv, else comma
    delimiter = '\t' if filepath.endswith('.tsv') else ','

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)  # skip header

        for row in reader:
            if len(row) < 3:
                continue
            head, relation, tail = row[0].strip(), row[1].strip(), row[2].strip()
            if head and tail:
                triples.append((head.lower(), relation.lower(), tail.lower()))
                count += 1
                if max_triples and count >= max_triples:
                    break

    print(f"[ATOMIC] Loaded {len(triples)} triples from {filepath}")
    return triples
