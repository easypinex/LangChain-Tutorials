"""整理建構圖樹方法"""
import os
from typing import Any, Dict, List
from uuid import uuid4 as uuid

from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import (Document, GraphDocument,
                                                       Node, Relationship)
# from langchain_core.documents.transformers import BaseDocumentTransformer


class TwlfGraphBuilder:
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self._tag_node_id_map = {}  # tag, node_id, 用以記憶每個tag node 的 id, 減少查詢

    def graph_build(self, doc_pages: List[Document], spliter, tags: List[str] | None = None):
        '''
        自動建立圖樹
        '''
        if len(doc_pages) == 0:
            return
        if spliter is None:
            raise ValueError('split_texts is empty')
        if tags is None:
            tags = []
        # filename: document_node
        graph_docs: List[GraphDocument] = []
        document = {}  # keys [document, node]
        pre_node = None
        graph_doc: GraphDocument = None
        doc_properties = {}
        if len(doc_pages) > 0:
            page = doc_pages[0]
            path = page.metadata['source']
            filename = os.path.basename(path)
            doc_properties = {
                'filename': filename,
                'file_path': path,
                'total_page_num': len(doc_pages)
            }
            document['node'] = Node(id=str(uuid()), type='Document')
            document['document'] = Document(page_content="")
            pre_node = document['node']
            graph_doc = GraphDocument(
                nodes=[], relationships=[], source=document['document'])
            graph_docs.append(graph_doc)
            for tag in tags:
                tag_node = self._get_tagnode(tag)
                self._tag_node_id_map[tag] = tag_node.id
                graph_doc.nodes.append(tag_node)
                tag_relationship = Relationship(
                    source=document['node'], target=tag_node, type='TAG')
                graph_doc.relationships.append(tag_relationship)

        chunk_idx = 0
        for page_idx, page in enumerate(doc_pages):
            page_content = page.page_content
            split_texts = spliter.split_text(page_content)
            for text in split_texts:
                properties = {
                    # 'chunk_idx': chunk_idx,
                    'content': text,
                    # 'name': f'Chunk_{chunk_idx}',
                    # 'source_idx': document['node'].id,
                    'page_num': page_idx + 1
                }
                chunk_idx += 1
                chunk_node = Node(id=str(uuid()), type='Chunk',
                                  properties=properties)
                graph_doc.nodes.append(chunk_node)
                relationship = Relationship(
                    source=pre_node, target=chunk_node, type='NEXT')
                relationship_part = Relationship(
                    source=document['node'], target=chunk_node, type='PART')
                graph_doc.relationships.append(relationship)
                graph_doc.relationships.append(relationship_part)
                pre_node = chunk_node

        self.graph.add_graph_documents(graph_docs)
        self._update_node_properties(document['node'].id, doc_properties)

    def _get_tagnode(self, tag_name):
        tag_properties = {
            'name': tag_name
        }
        if self._tag_node_id_map.get(tag_name) is not None:
            tag_node = Node(id=self._tag_node_id_map.get(
                tag_name), type='Tag', properties=tag_properties)
            return tag_node
        tag_query = f"""
            MATCH (n:Tag {{name: '{tag_name}'}})
            RETURN n.id as id
        """
        query_results = self.graph.query(tag_query)
        if len(query_results) == 0:
            tag_node = Node(id=str(uuid()), type='Tag',
                            properties=tag_properties)
            return tag_node
        tag_dict = query_results[0]
        tag_node = Node(id=tag_dict['id'], type='Tag',
                        properties=tag_properties)
        return tag_node

    def _update_node_properties(self, node_id: str, node_properties: dict) -> List[Dict[str, Any]] | None:
        if len(node_properties) == 0:
            return None
        set_query = ''
        for key in node_properties.keys():
            set_query += f'n.{key} = ${key}, '
        set_query = set_query[:-2]
        temp = f'''
                MATCH (n) WHERE n.id = '{node_id}'
                SET {set_query}
                RETURN n
                '''
        return self.graph.query(temp, node_properties)
