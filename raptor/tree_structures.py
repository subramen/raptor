from typing import Dict, List, Set, Optional
from dataclasses import dataclass


@dataclass
class Node:
    """
    Represents a node in the hierarchical tree structure.
    """
    text: str
    index: int
    children: Set[int]
    embeddings: Optional[List] = None

@dataclass
class Tree:
    """
    Represents the entire hierarchical tree structure.
    """
    all_nodes: List
    root_nodes: List
    leaf_nodes: List
    num_layers: int
    layer_to_nodes: Dict

@dataclass
class Document:
    source_filepath: str
    tree: Tree = None
    metadata: Dict = None

    def get_text(self):
        with open(self.source_filepath) as f:
            text = f.read()
        return text
