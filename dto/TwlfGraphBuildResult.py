from typing import List, TypedDict

from langchain_community.graphs.graph_document import (Document)

from .GraphGraphDetails import GraphGraphDetails

class ChunkList(TypedDict):
    chunk_id: str
    chunk_doc: Document
    
class TwlfGraphBuildResult:
    def __init__(self, chunks: List[ChunkList] = None, details: List[GraphGraphDetails] = None):
        self.chunks: List[ChunkList] = chunks
        if details == None:
            details = []
        self.details: List[GraphGraphDetails] = details

