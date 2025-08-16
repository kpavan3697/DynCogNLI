"""
retriever.py

This module is the core component for retrieving and combining common sense knowledge from
multiple sources. It's designed to interface with two distinct knowledge graphs, ConceptNet 
and ATOMIC, and integrate their information into a single, cohesive graph structure. The
module performs **concept extraction** from natural language queries, **matches** these
concepts to nodes in the knowledge graphs, and then performs a **subgraph retrieval**
operation to gather relevant, multi-hop context. This combined subgraph serves as the
input for downstream reasoning tasks, such as those performed by a Graph Neural Network (GNN).
"""

import spacy
import networkx as nx
from typing import Set, List, Tuple, Dict, Any, Union

# Load spaCy model for Natural Language Processing.
try:
    # A small, lightweight English model for basic NLP tasks.
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not found, download it automatically.
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_concepts(text: str) -> Set[str]:
    """
    Extracts key concepts from a given text using spaCy.

    This function identifies important words and phrases by filtering for nouns,
    proper nouns, and named entities, while excluding common stopwords and
    short, non-alphabetic tokens. It also lemmatizes words to their base form.

    Args:
        text (str): The input text query.

    Returns:
        Set[str]: A set of unique, lemmatized, and lowercased key concepts.
    """
    doc = nlp(text)
    concepts: Set[str] = set()

    # Iterate through tokens to find relevant parts of speech.
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and token.is_alpha and len(token.text) > 2:
            concepts.add(token.lemma_.lower())
    
    # Iterate through named entities and add them, filtering out numeric types.
    for ent in doc.ents:
        if ent.label_ not in ["CARDINAL", "ORDINAL", "DATE", "TIME", "MONEY", "QUANTITY", "PERCENT"]:
            concepts.add(ent.text.lower())
    
    return concepts

def retrieve_combined_subgraph(
    conceptnet_graph: nx.Graph,
    atomic_graph: nx.Graph,
    input_data: Union[str, List[str]],
    k: int = 2,
    is_concept_list: bool = False
) -> nx.Graph:
    """
    Retrieves and merges relevant subgraphs from ConceptNet and ATOMIC.

    This function retrieves a combined subgraph by first identifying key concepts
    from the input (either a text query or a list of pre-matched concepts). It then
    traverses both the ConceptNet and ATOMIC graphs, starting from the matched
    concept nodes and expanding outwards for `k` hops. The resulting subgraphs are
    then merged into a single `NetworkX` graph.

    Args:
        conceptnet_graph (nx.Graph): The pre-loaded ConceptNet knowledge graph.
        atomic_graph (nx.Graph): The pre-loaded ATOMIC knowledge graph.
        input_data (Union[str, List[str]]): The user query string or a list of
                                            matched concept nodes.
        k (int): The number of hops to traverse from the starting concepts.
                 Defaults to 2.
        is_concept_list (bool): Flag to indicate if `input_data` is a list of
                                concepts instead of a raw query string.

    Returns:
        nx.Graph: A single, merged NetworkX graph containing the relevant
                  subgraphs from both knowledge bases.
    """
    # Determine the list of concepts to search for in the graphs.
    if is_concept_list:
        matched_concepts = input_data
    else:
        # This branch is for demonstration and not the primary use case
        # as per the module comments.
        concepts = extract_concepts(input_data)
        # Assuming a `normalize_and_match_concepts` utility exists elsewhere.
        matched_concepts = concepts

    subgraph = nx.Graph()

    # Retrieve and merge subgraph from ConceptNet.
    for concept in matched_concepts:
        if concept in conceptnet_graph:
            nodes_in_range = nx.single_source_shortest_path_length(conceptnet_graph, concept, cutoff=k).keys()
            
            # Add nodes and their attributes.
            for node in nodes_in_range:
                if node in conceptnet_graph:
                    subgraph.add_node(node, **conceptnet_graph.nodes[node])
            
            # Add edges and their attributes.
            for u, v, data in conceptnet_graph.edges(nodes_in_range, data=True):
                if u in nodes_in_range and v in nodes_in_range:
                    subgraph.add_edge(u, v, **data)

    # Retrieve and merge subgraph from ATOMIC.
    # Note: ATOMIC focuses on social commonsense, e.g., "if-then" relations.
    for concept in matched_concepts:
        if concept in atomic_graph:
            nodes_in_range = nx.single_source_shortest_path_length(atomic_graph, concept, cutoff=k).keys()
            
            for node in nodes_in_range:
                if node in atomic_graph:
                    subgraph.add_node(node, **atomic_graph.nodes[node])
            
            for u, v, data in atomic_graph.edges(nodes_in_range, data=True):
                if u in nodes_in_range and v in nodes_in_range:
                    subgraph.add_edge(u, v, **data)

    return subgraph

# Utility functions (might be defined in other modules in a real project).
def normalize_conceptnet_node(concept: str) -> str:
    """Normalizes a concept string to match ConceptNet's node format."""
    return concept.lower()

# The `retrieve_subgraph` function is provided as a reference but is not
# the primary retrieval mechanism for combining graphs in this module.