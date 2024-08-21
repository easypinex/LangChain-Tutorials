"""整理建構圖樹方法"""
import os
from typing import Any, Dict, List
from uuid import uuid4 as uuid

from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import (Document, GraphDocument,
                                                       Node, Relationship)
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents.transformers import BaseDocumentTransformer
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from langchain_experimental.graph_transformers import LLMGraphTransformer


class TwlfGraphBuilder:
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self._tag_node_id_map = {}  # tag, node_id, 用以記憶每個tag node 的 id, 減少查詢
        self.chunk_docs: List[Document] = []
        self.chunk_list: List[dict] = []
        self.graph_document: GraphDocument | None = None

    def graph_build(self, doc_pages: List[Document], spliter=None, tags: List[str] | None = None):
        '''
        自動建立圖樹
        '''
        if len(doc_pages) == 0:
            return
        if spliter is None:
            spliter = RecursiveCharacterTextSplitter(
                chunk_size=300, chunk_overlap=30, separators=['\n\n', '，', '。', '【', ','])
        if tags is None:
            tags = []
        # filename: document_node
        document = {}  # keys [document, node]
        pre_node = None
        self.graph_document: GraphDocument = None
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
        self.graph_document = GraphDocument(
            nodes=[], relationships=[], source=document['document'])
        for tag in tags:
            tag_node = self._get_tagnode(tag)
            self._tag_node_id_map[tag] = tag_node.id
            self.graph_document.nodes.append(tag_node)
            tag_relationship = Relationship(
                source=document['node'], target=tag_node, type='TAG')
            self.graph_document.relationships.append(tag_relationship)

        for page_idx, page in enumerate(doc_pages):
            page_content = page.page_content
            split_texts = spliter.split_text(page_content)
            for text in split_texts:
                text = self._bad_chars_clear(text)
                properties = {
                    'content': text,
                }
                metadata = {
                    'source': page.metadata['source'],
                    'page_number': page_idx + 1
                }

                chunk_node = Node(id=str(uuid()), type='Chunk',
                                  properties=properties)
                chunk_doc = Document(page_content=text, metadata=metadata)
                self.chunk_list.append(
                    {'chunk_id': chunk_node.id, 'chunk_doc': chunk_doc})
                self.chunk_docs.append(chunk_doc)
                self.graph_document.nodes.append(chunk_node)
                relationship = Relationship(
                    source=pre_node, target=chunk_node, type='NEXT')
                relationship_part = Relationship(
                    source=document['node'], target=chunk_node, type='PART')
                self.graph_document.relationships.append(relationship)
                self.graph_document.relationships.append(relationship_part)
                pre_node = chunk_node

        self.graph.add_graph_documents([self.graph_document])
        self._update_node_properties(document['node'].id, doc_properties)
        return self.graph_document

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

    def _bad_chars_clear(self, text="", bad_chars: List[str] | None = None):
        if bad_chars is None:
            bad_chars = ['"', "\n", "'"]
        for bad_char in bad_chars:
            if bad_char == '\n':
                text = text.replace(bad_char, ' ')
            else:
                text = text.replace(bad_char, '')
        return text

    def get_chunk_and_graphDocument(self, graph_document_list: List[GraphDocument]) -> List[dict]:
        '''
        將圖樹中的 chunk 與 節點 提取出來, 提供陣列

        params:
            graph_document_list: 圖陣列
        return:
            lst_chunk_chunkId_document: [{'graph_doc': GraphDocument, 'chunk_id': str}, ...]
        '''
        logging.info(
            "creating list of chunks and graph documents in get_chunk_and_graphDocument func")
        lst_chunk_chunkId_document = []
        for graph_document in graph_document_list:
            for chunk_id in graph_document.source.metadata['combined_chunk_ids']:
                lst_chunk_chunkId_document.append(
                    {'graph_doc': graph_document, 'chunk_id': chunk_id})

        return lst_chunk_chunkId_document

    def merge_relationship_between_chunk_and_entites(self, graph_documents_chunk_chunk_Id: list) -> List[dict]:
        '''
        將 chunk 與 節點 的關係於資料庫中串在一起

        params:
            graph_documents_chunk_chunk_Id: [{'graph_doc': GraphDocument, 'chunk_id': str}, ...]
        '''
        batch_data = []
        logging.info(
            "Create HAS_ENTITY relationship between chunks and entities")
        chunk_node_id_set = 'id:"{}"'
        for graph_doc_chunk_id in graph_documents_chunk_chunk_Id:
            for node in graph_doc_chunk_id['graph_doc'].nodes:
                query_data = {
                    'chunk_id': graph_doc_chunk_id['chunk_id'],
                    'node_type': node.type,
                    'node_id': node.id
                }
                batch_data.append(query_data)
                # node_id = node.id
                # Below query is also unable to change as parametrize because we can't make parameter of Label or node type
                # https://neo4j.com/docs/cypher-manual/current/syntax/parameters/
                # graph.query('MATCH(c:Chunk {'+chunk_node_id_set.format(graph_doc_chunk_id['chunk_id'])+'}) MERGE (n:'+ node.type +'{ id: "'+node_id+'"}) MERGE (c)-[:HAS_ENTITY]->(n)')

        if batch_data:
            unwind_query = """
                        UNWIND $batch_data AS data
                        MATCH (c:Chunk {id: data.chunk_id})
                        CALL apoc.merge.node([data.node_type], {id: data.node_id}) YIELD node AS n
                        MERGE (c)-[:HAS_ENTITY]->(n)
                    """
            self.graph.query(unwind_query, params={"batch_data": batch_data})

    def get_graph_from_llm(self, llm, chunkId_chunkDoc_list, allowedNodes, allowedRelationship) -> List[GraphDocument]:
        '''
        由 LLM 取得圖樹

        params:
            llm: LLM 模型
            chunkId_chunkDoc_list: [{'chunk_id': str, 'chunk_doc': Document}, ...]
            allowedNodes: List[str] 允許的節點類型(Label)
            allowedRelationship: List[str] 允許的關係
            
        return:
            graph_document_list: List[GraphDocument]
        '''
        combined_chunk_document_list = self._get_combined_chunks(
            chunkId_chunkDoc_list)
        graph_document_list = self._get_graph_document_list(
            llm, combined_chunk_document_list, allowedNodes, allowedRelationship
        )
        return graph_document_list

    def _get_combined_chunks(self, chunkId_chunkDoc_list, chunks_to_combine=1):
        logging.info(
            f"Combining {chunks_to_combine} chunks before sending request to LLM")
        combined_chunk_document_list = []
        combined_chunks_page_content = [
            "".join(
                document["chunk_doc"].page_content
                for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
            )
            for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
        ]
        combined_chunks_ids = [
            [
                document["chunk_id"]
                for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
            ]
            for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
        ]

        for i in range(len(combined_chunks_page_content)):
            combined_chunk_document_list.append(
                Document(
                    page_content=combined_chunks_page_content[i],
                    metadata={"combined_chunk_ids": combined_chunks_ids[i]},
                )
            )
        return combined_chunk_document_list

    def _get_graph_document_list(
        self, llm, combined_chunk_document_list, allowedNodes, allowedRelationship
    ):
        futures = []
        graph_document_list = []
        if llm.get_name() == "ChatOllama":
            node_properties = False
        else:
            node_properties = ["description"]
        llm_transformer = LLMGraphTransformer(
            llm=llm,
            node_properties=node_properties,
            allowed_nodes=allowedNodes,
            allowed_relationships=allowedRelationship,
        )
        with ThreadPoolExecutor(max_workers=10) as executor:
            for chunk in combined_chunk_document_list:
                chunk_doc = Document(
                    page_content=chunk.page_content.encode("utf-8"), metadata=chunk.metadata
                )
                futures.append(
                    executor.submit(
                        llm_transformer.convert_to_graph_documents, [chunk_doc])
                )

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                graph_document = future.result()
                graph_document_list.append(graph_document[0])

        return graph_document_list
