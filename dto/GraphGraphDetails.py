from langchain_community.graphs.graph_document import (Document, GraphDocument,
                                                       Node, Relationship)
class GraphGraphDetails:
    def __init__(self, root_document: Document, root_node: Node, root_graph_document: GraphDocument):
        self.root_document = root_document
        self.root_node = root_node
        self.root_graph_document = root_graph_document