"""
graph_utils.py

This module provides a set of utility functions for manipulating and analyzing
knowledge graphs. These functions are designed to support the graph-based
reasoning pipeline by handling tasks such as:
- Finding approximate matches for concepts in a graph.
- Normalizing and matching user-provided concepts to nodes in the graph.
- Merging multiple knowledge graph data sources into a unified structure.
"""
import difflib
import networkx as nx
from typing import List, Dict, Any, Union, Iterable, Optional

def find_closest_node(graph_nodes: Iterable[str], concept: str, cutoff: float = 0.8) -> List[str]:
    """
    Finds the closest string matches for a given concept within a list of graph nodes.

    This function uses Python's `difflib` library for fuzzy string matching, which is
    useful for handling slight variations or typos in user-provided concepts. It
    returns a list of potential matches based on a similarity `cutoff`.

    Args:
        graph_nodes (Iterable[str]): A list or iterable of node names to search within.
        concept (str): The concept string to find a match for.
        cutoff (float): The minimum similarity score (from 0.0 to 1.0) required
                        for a match to be considered.

    Returns:
        List[str]: A list of strings from `graph_nodes` that are the closest
                   matches to the input `concept` and meet the cutoff.
    """
    return difflib.get_close_matches(concept, graph_nodes, n=3, cutoff=cutoff)

def normalize_and_match_concepts(graph: nx.Graph, concepts: List[str]) -> Dict[str, Optional[str]]:
    """
    Normalizes concepts and finds the best match within a given graph.

    For each input concept, this function first checks for an exact, case-insensitive
    match in the graph's nodes. If an exact match is not found, it performs a fuzzy
    match using `find_closest_node`. This helps to robustly link user queries to the
    nodes in the knowledge graph.

    Args:
        graph (nx.Graph): The NetworkX graph to search within.
        concepts (List[str]): A list of concept strings to normalize and match.

    Returns:
        Dict[str, Optional[str]]: A dictionary mapping each original concept to its
                                  best matched node name from the graph, or `None`
                                  if no suitable match was found.
    """
    graph_nodes_lower = {n.lower(): n for n in graph.nodes()}
    concept_matches: Dict[str, Optional[str]] = {}

    for concept in concepts:
        concept_lower = concept.lower()
        if concept_lower in graph_nodes_lower:
            # Found an exact match (case-insensitive)
            matched_node = graph_nodes_lower[concept_lower]
            concept_matches[concept] = matched_node
        else:
            # Try fuzzy matching on the lowercased graph nodes
            close_matches_lower = find_closest_node(list(graph_nodes_lower.keys()), concept_lower)
            if close_matches_lower:
                # Pick the best match (the first in the list) and map back to original casing
                best_match_lower = close_matches_lower[0]
                matched_node = graph_nodes_lower[best_match_lower]
                concept_matches[concept] = matched_node
            else:
                concept_matches[concept] = None

    return concept_matches

def merge_kg(
    conceptnet_triples: Union[List[Tuple[str, str, str]], Iterable[Tuple[str, str, str]]],
    atomic_triples: Union[List[Tuple[str, str, str]], Iterable[Tuple[str, str, str]]]
) -> nx.MultiDiGraph:
    """
    Merges knowledge triples from ConceptNet and ATOMIC into a single unified
    NetworkX MultiDiGraph.

    A `MultiDiGraph` is used to allow for multiple edges between the same two
    nodes, which is necessary if both ConceptNet and ATOMIC contain a relation
    between the same two concepts.

    Args:
        conceptnet_triples: An iterable of ConceptNet triples.
        atomic_triples: An iterable of ATOMIC triples.

    Returns:
        nx.MultiDiGraph: A unified knowledge graph containing all triples from
                         both sources.
    """
    G = nx.MultiDiGraph()
    for h, r, t in conceptnet_triples:
        G.add_edge(h, t, relation=r, source="conceptnet")
    for h, r, t in atomic_triples:
        G.add_edge(h, t, relation=r, source="atomic")
    return G
