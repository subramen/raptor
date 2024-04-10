from neomodel import StructuredNode, StringProperty, IntegerProperty, ArrayProperty, RelationshipTo, config, RelationshipFrom
from tree_structures import Node, Tree

class NeoDoc(StructuredNode):
    name = StringProperty(required=True)
    filepath = StringProperty()
    contains = RelationshipTo('NeoNode', 'CONTAINS')


class NeoNode(StructuredNode):
    """
    Represents a leaf node in the hierarchical tree structure.
    """
    text = StringProperty(required=True)
    # span = ArrayProperty(IntegerProperty(), required=True)
    layer = IntegerProperty()
    embeddings = ArrayProperty()
    is_summary_of = RelationshipTo('NeoNode', 'IS_SUMMARY_OF')
    contained_in = RelationshipTo('NeoDoc', 'CONTAINED_IN')  # should span be included as a property here?


def retrieve_tree_from_neomodel(document_name):
    query = f"""
        Match (n:NeoNode)-[:HAS_SOURCE]->(d: NeoDoc)
        Optional MATCH (m:NeoNode)-[:HAS_SUMMARY]->(n)
        where d.filepath="{document_name}"
        RETURN n as node, collect(elementId(m)) as children
        ORDER BY elementId(n);
        """
    resultset, result_keys = db.cypher_query(query, resolve_objects=True)
    layer_to_nodes = {}
    all_nodes = {}
    root_nodes = {}
    leaf_nodes = {}

    for neo_node, children in resultset:
        children = children.pop()
        node = Node(text=neo_node.text, embeddings=neo_node.embeddings, span=neo_node.span, index=int(neo_node.element_id.split(':')[-1]), children=children)
        all_nodes[node.index] = node
        layer_to_nodes.get(neo_node.layer, []).append(node)

    num_layers = max(layer_to_nodes.keys())

    for n in layer_to_nodes[0]:
        leaf_nodes[n.index] = n

    for n in layer_to_nodes[num_layers]:
        root_nodes[n.index] = n

    return Tree(all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes)



def create_tree_in_neomodel(tree, document_name, document_filepath, uri, user, password):
    """
    Create a tree in Neo4J from a Python Tree object.
    :param tree: The Python Tree object.
    :param uri: The URI of the Neo4J instance.
    :param user: The username to connect to the Neo4J instance.
    :param password: The password to connect to the Neo4J instance.
    """
    # Connect to the Neo4J instance
    config.DATABASE_URL = f'neo4j+s://{user}:{password}@{uri}'
    # Create a node for the source Document
    doc_node = NeoDoc(filepath=document_filepath, name=document_name).save()
    # Create a dictionary to store the Neo4J nodes
    neo_nodes = {}
    # Create all nodes in Neo4J
    for layer, layer_nodes in tree.layer_to_nodes.items():
        for node in layer_nodes:
            neo_node = NeoNode(text=node.text, layer=layer, embeddings=node.embeddings['JinaAI']).save()
            neo_nodes[node.index] = neo_node
            if layer == 0:
                neo_node.contained_in.connect(doc_node)
                doc_node.contains.connect(neo_node)
    # Create the relationships in Neo4J
    for node in tree.all_nodes:
        # relationship with children (summary) nodes
        for child_index in node.children:
            child_neo_node = neo_nodes[child_index]
            neo_nodes[node.index].is_summary_of.connect(child_neo_node)
