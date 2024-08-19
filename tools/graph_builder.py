"""整理建構圖樹方法"""
import os
from typing import List
from uuid import uuid4 as uuid

from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import (Document, GraphDocument,
                                                       Node, Relationship)
# from langchain_core.documents.transformers import BaseDocumentTransformer


def graph_build(graph: Neo4jGraph, doc_pages: List[Document], spliter):
    '''
    自動建立圖樹
    '''
    # filename: document_node
    graph_docs: List[GraphDocument] = []
    document = {} # keys [document, node]
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
        graph_doc = GraphDocument(nodes=[], relationships=[], source=document['document'])
        graph_docs.append(graph_doc)

    chunk_idx = 0
    for page_idx, page in enumerate(doc_pages):
        page_content = page.page_content
        split_texts = spliter.split_text(page_content)
        for text in split_texts:
            properties = {
                'chunk_idx': chunk_idx,
                'content': text,
                'name': f'Chunk_{chunk_idx}',
                'source_idx': document['node'].id,
                'page_num': page_idx + 1
            }
            chunk_idx+=1
            chunk_node = Node(id=str(uuid()), type='Chunk', properties=properties)
            graph_doc.nodes.append(chunk_node)
            relationship = Relationship(source=pre_node, target=chunk_node, type='Next')
            relationship_part = Relationship(source=document['node'], target=chunk_node, type='Part')
            graph_doc.relationships.append(relationship)
            graph_doc.relationships.append(relationship_part)
            pre_node = chunk_node
            
    graph.add_graph_documents(graph_docs)
    set_query = ''
    for key, item in doc_properties.items():
        quote = "'" if isinstance(item, str) else ""
        set_query += f'n.{key} = {quote}{item}{quote}, '
    set_query = set_query[:-2]
    temp = f'''
            MATCH (n) WHERE n.id = '{document['node'].id}'
            SET {set_query}
            RETURN n
            '''
    graph.query(temp)
    