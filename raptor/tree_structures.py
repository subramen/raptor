from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Node:
    """
    Represents a node in the hierarchical tree structure.
    """
    text: str
    index: int
    children: Set[int]
    span: Tuple[int, int] = (-1, -1,)
    embeddings: Dict[str, List] = None

@dataclass
class Tree:
    """
    Represents the entire hierarchical tree structure.
    """
    all_nodes: Dict[int, Node]
    root_nodes: Dict[int, Node]
    leaf_nodes: Dict[int, Node]
    num_layers: int
    layer_to_nodes: Dict[int, List[Node]]

@dataclass
class Document:
    source_filepath: str

    def get_text(self):
        with open(source_filepath) as f:
            text = f.read()
        return text
