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

from tools.TWLF_LLMGraphTransformer import TWLF_LLMGraphTransformer

from tqdm import tqdm


class TwlfGraphBuilder:
    def __init__(self, graph: Neo4jGraph, max_thread=10):
        self.graph = graph
        self._tag_node_id_map = {}  # tag, node_id, 用以記憶每個tag node 的 id, 減少查詢
        self._max_thread = max_thread
        self._source_doc_map = {}

    def graph_build(self, docs: List[Document], spliter=None):
        """_summary_

        Args:
            docs (List[Document]): 所有的Document文檔
            spliter (optional): LangChain任意Spliter皆可, 只要有 split_text(str)即可. Defaults to None.
            tags (List[str] | None, optional): _description_. Defaults to None.

        Returns:
            chunks (List[dict]): 
                {'chunk_id': uuid,'chunk_doc': Document }
        """        
        if len(docs) == 0:
            return
        if spliter is None:
            spliter = RecursiveCharacterTextSplitter(
                chunk_size=300, chunk_overlap=30, separators=['\n\n', '，', '。', '【', ','])
        chunks = [] # final result
        for doc in docs:
            doc_metadata = doc.metadata
            source = None
            if doc_metadata is not None and 'source' in doc_metadata:
                source = doc_metadata['source']
            document_dict = self._get_source_document(source)
            graph_document = document_dict['graph_document']
            document = document_dict['document']
            page_content = doc.page_content
            split_texts = spliter.split_text(page_content)
            for text in split_texts:
                pre_node = document_dict['pre_node']
                text = self._bad_chars_clear(text)
                properties = {
                    'source': source,
                    'content': text,
                    'page_number':  document.metadata['total_page_num']
                }
                chunk_node = Node(id=str(uuid()), type='__Chunk__',
                                    properties=properties)
                chunk_doc = Document(page_content=text, metadata=properties)
                chunks.append(
                    {'chunk_id': chunk_node.id, 'chunk_doc': chunk_doc})
                graph_document.nodes.append(chunk_node)
                relationship = Relationship(
                    source=pre_node, target=chunk_node, type='NEXT')
                relationship_part = Relationship(
                    source=document_dict['node'], target=chunk_node, type='PART')
                graph_document.relationships.append(relationship)
                graph_document.relationships.append(relationship_part)
                document_dict['pre_node'] = chunk_node
            self.graph.add_graph_documents([graph_document])
            self._update_node_properties(document_dict['node'].id, document_dict['document'].metadata)
        return chunks

    def _get_source_document(self, source: str):
        """
        從 source(檔案路徑) 取得 document
        Args:
            source (str): 檔案名稱 , 或任何 String

        Returns:
            document (dict): 
                {'document': Document, 'node': Node, 'page': int, 'graph_document': GraphDocument, 'pre_node': Document}
        """        
        if source in self._source_doc_map:
            self._source_doc_map[source]['document'].metadata['total_page_num'] += 1
            return self._source_doc_map[source]
        
        document_dict = {}  # keys [document, node]
        document_dict['node'] = Node(id=str(uuid()), type='__Document__')
        filename = os.path.basename(source)
        doc_properties = {
            'id': document_dict['node'].id,
            'filename': filename,
            'file_path': source,
            'total_page_num': 1,
        }
        document_dict['document'] = Document(page_content="", metadata=doc_properties)
        graph_document = GraphDocument(
            nodes=[], relationships=[], source=document_dict['document'])
        document_dict['graph_document'] = graph_document
        document_dict['pre_node'] = document_dict['node']
        self._source_doc_map[source] = document_dict
        return document_dict

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
            bad_chars = ['"', "\n", "'", "..."]
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
                        MATCH (c:__Chunk__ {id: data.chunk_id})
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
            llm, combined_chunk_document_list, allowedNodes, allowedRelationship, max_retry=2
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
        self, llm, combined_chunk_document_list, allowedNodes, allowedRelationship, max_retry=0
    ):
        
        futures = []
        graph_document_list = []
        node_properties = ["description"]
        llm_transformer = TWLF_LLMGraphTransformer(
            llm=llm,
            node_properties=node_properties,
            allowed_nodes=allowedNodes,
            allowed_relationships=allowedRelationship,
        )
        futures_to_chunk_doc = {}
        failed_documents = []
        with ThreadPoolExecutor(max_workers=self._max_thread) as executor:
            for chunk in combined_chunk_document_list:
                chunk_doc = Document(
                    page_content=chunk.page_content.encode("utf-8"), metadata=chunk.metadata
                )
                future = executor.submit(
                    llm_transformer.convert_to_graph_documents, [chunk_doc]
                )
                futures.append(future)
                futures_to_chunk_doc[future] = chunk  # 關聯 future 和 chunk_doc    

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(combined_chunk_document_list)):
                try:
                    graph_document = future.result(timeout=5 * 60) # 一個Chunk最多等待5分鐘
                    graph_document_list.append(graph_document[0])
                except Exception as e:
                    chunk = futures_to_chunk_doc[future]
                    failed_documents.append(chunk)
                    print(f"Error processing document: {chunk}")
                    print(e)
        if len(failed_documents) > 0 and max_retry > 0:
            graph_document_list += self._get_graph_document_list(llm, failed_documents, allowedNodes, allowedRelationship, max_retry-1)
        return graph_document_list
