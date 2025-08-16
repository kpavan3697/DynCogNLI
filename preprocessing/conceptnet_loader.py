"""
conceptnet_loader.py

This module provides a function for loading knowledge triples from ConceptNet,
specifically designed to handle both standard CSV and gzipped CSV files. It
extracts English-language triples and normalizes them into a consistent
(head, relation, tail) format. This utility is essential for ingesting a
large-scale commonsense knowledge base to be used in systems like DynCogNLI.
"""
import gzip
import os
import re
from typing import List, Tuple, Optional

# Regular expression to find English concepts in a ConceptNet URI.
# For example, it extracts "dog" from "/c/en/dog".
_CONCEPTNET_EN_PATTERN = re.compile(r'/c/en/([^/,\s]+)', re.IGNORECASE)

# Regular expression to find the relationship in a ConceptNet URI.
# For example, it extracts "RelatedTo" from "/r/RelatedTo".
_REL_PATTERN = re.compile(r'/r/([^/,\s]+)', re.IGNORECASE)

def load_conceptnet(path: str, max_triples: Optional[int] = None) -> List[Tuple[str, str, str]]:
    """
    Loads knowledge triples from a ConceptNet CSV or CSV.GZ file.

    This function reads a ConceptNet file, processes each line, and extracts
    knowledge triples where both the head and tail are in English. It uses
    regular expressions to parse the URIs, and then cleans the resulting
    concepts and relations by replacing underscores with spaces and converting
    them to lowercase for consistent representation.

    Args:
        path (str): The file path to the ConceptNet data, which can be a `.csv`
                    or a `.csv.gz` file. The function automatically handles
                    compressed files.
        max_triples (Optional[int]): An optional limit on the number of triples to load.
                                     This is useful for debugging. If None, all triples
                                     are loaded.

    Returns:
        List[Tuple[str, str, str]]: A list of tuples, each representing an
                                     English knowledge triple in the format
                                     (head, relation, tail). Returns an empty
                                     list if the file is not found.
    """
    if not os.path.exists(path):
        print(f"[ConceptNet] Path not found: {path}")
        return []

    # Use gzip.open for .gz files and standard open for others.
    # The 'rt' mode is for reading text, which is necessary for gzip.
    open_fn = gzip.open if path.endswith('.gz') else open
    triples = []
    count = 0

    try:
        with open_fn(path, 'rt', encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                # Find all English concepts in the line. ConceptNet often has
                # multilingual data, so this ensures we only get English triples.
                nodes = _CONCEPTNET_EN_PATTERN.findall(line)
                
                if len(nodes) >= 2:
                    # Extract and normalize the head and tail concepts.
                    h = nodes[0].replace('_', ' ').strip().lower()
                    t = nodes[1].replace('_', ' ').strip().lower()

                    # Extract the relation. If not found, default to "related".
                    rel_m = _REL_PATTERN.search(line)
                    r = rel_m.group(1).replace('_', ' ').strip().lower() if rel_m else "related"

                    triples.append((h, r, t))
                    count += 1

                    if max_triples is not None and count >= max_triples:
                        break

    except Exception as e:
        print(f"[ConceptNet] An error occurred while reading the file: {e}")
        return []

    print(f"[ConceptNet] Loaded {len(triples)} triples from {path}")
    return triples
