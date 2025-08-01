#preprocessing/atomic_loader.py
"""
atomic_loader.py

Loads and preprocesses ATOMIC commonsense knowledge data for use in graph construction and reasoning.
Handles parsing, filtering, and formatting of ATOMIC triples.
"""
import csv

def load_atomic_tsv(filepath):
    triples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)  # Skip header if it exists
        for row in reader:
            if len(row) < 3:
                continue
            head, relation, tail = row[0], row[1], row[2]
            triples.append((head, relation, tail))
    return triples