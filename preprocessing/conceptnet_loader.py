# preprocessing/conceptnet_loader.py
import gzip
import os
import re

_CONCEPTNET_EN_PATTERN = re.compile(r'/c/en/([^/,\s]+)', re.IGNORECASE)
_REL_PATTERN = re.compile(r'/r/([^/,\s]+)', re.IGNORECASE)

def load_conceptnet(path, max_triples=None):
    """
    Load ConceptNet from .csv or .csv.gz.
    Extract only English triples as (head, relation, tail) in lowercase.
    """
    if not os.path.exists(path):
        print(f"[ConceptNet] Path not found: {path}")
        return []

    open_fn = gzip.open if path.endswith('.gz') else open
    triples = []
    count = 0

    with open_fn(path, 'rt', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            nodes = _CONCEPTNET_EN_PATTERN.findall(line)
            if len(nodes) >= 2:
                h = nodes[0].replace('_', ' ').strip().lower()
                t = nodes[1].replace('_', ' ').strip().lower()
                rel_m = _REL_PATTERN.search(line)
                r = rel_m.group(1).replace('_', ' ').strip().lower() if rel_m else "related"

                triples.append((h, r, t))
                count += 1

                if max_triples is not None and count >= max_triples:
                    break

    print(f"[ConceptNet] Loaded {len(triples)} triples from {path}")
    return triples
