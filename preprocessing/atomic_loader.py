"""
atomic_loader.py

This module provides a robust function for loading knowledge triples from ATOMIC-style
TSV or CSV files. It is designed to be a core utility for any system that needs to
ingest and process structured commonsense knowledge data, such as the DynCogNLI system.
The module handles file I/O, delimiter detection, and data normalization (stripping
whitespace and converting to lowercase) to ensure clean data for further processing.
"""
import os
import csv
from typing import List, Tuple, Optional

def load_atomic_tsv(filepath: str, max_triples: Optional[int] = None) -> List[Tuple[str, str, str]]:
    """
    Loads ATOMIC-style knowledge triples from a TSV or CSV file.

    The function reads a file, automatically detects the delimiter (tab for .tsv, comma for .csv),
    and parses each line to extract a (head, relation, tail) triple. The extracted components
    are converted to lowercase and any leading/trailing whitespace is removed. This normalization
    is crucial for consistent data processing later in the pipeline.

    Args:
        filepath (str): The full path to the ATOMIC data file. The function supports both
                        `.tsv` and `.csv` extensions and determines the delimiter accordingly.
        max_triples (Optional[int]): An optional limit on the number of triples to load.
                                     This is useful for debugging or working with subsets of
                                     a large dataset. If None, all triples are loaded.

    Returns:
        List[Tuple[str, str, str]]: A list of tuples, where each tuple represents a knowledge
                                     triple in the format (head, relation, tail). An empty
                                     list is returned if the file cannot be found or read.
    """
    # Check if the file path exists to prevent a FileNotFoundError.
    if not os.path.exists(filepath):
        print(f"[ATOMIC] Path not found: {filepath}")
        return []

    triples = []
    count = 0

    # Auto-detect the delimiter based on the file extension.
    # This makes the function flexible for different file formats.
    delimiter = '\t' if filepath.endswith('.tsv') else ','

    try:
        # Use a `with` statement for safe and automatic file handling.
        # `errors='ignore'` is used to handle potential Unicode decoding issues, which are common
        # in large text datasets like ATOMIC.
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter=delimiter)
            
            # Skip the header row, which is standard practice for CSV/TSV data.
            header = next(reader, None)

            for row in reader:
                # Ensure the row has at least 3 columns to form a valid triple.
                if len(row) < 3:
                    continue
                
                # Extract the head, relation, and tail. Strip whitespace and convert to lowercase.
                head, relation, tail = row[0].strip(), row[1].strip(), row[2].strip()

                # Only append the triple if both head and tail are non-empty strings.
                if head and tail:
                    triples.append((head.lower(), relation.lower(), tail.lower()))
                    count += 1
                    
                    # Stop loading if the max_triples limit has been reached.
                    if max_triples and count >= max_triples:
                        break

    except Exception as e:
        # Catch any other potential file reading errors and print a helpful message.
        print(f"[ATOMIC] An error occurred while reading the file: {e}")
        return []

    print(f"[ATOMIC] Loaded {len(triples)} triples from {filepath}")
    return triples

