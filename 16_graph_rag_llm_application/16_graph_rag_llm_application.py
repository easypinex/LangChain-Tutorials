#!/usr/bin/env python
# coding: utf-8

# In[20]:


import sys
sys.path.append("..")


# In[21]:


import os

from langchain_community.graphs import Neo4jGraph

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "2wsx3edc"
database = os.environ.get('NEO4J_DATABASE')
graph = Neo4jGraph(database=database)


# In[22]:


from langchain_openai import AzureOpenAIEmbeddings

embedding = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint='https://sales-chatbot-llm.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15',
    azure_deployment='text-embedding-3-small',
    openai_api_version='2023-05-15'
)


# In[23]:


# from langchain_community.vectorstores import Neo4jVector
# # ! pip3 install -U langchain-huggingface
# import os
# os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/storage/models/embedding_models'
# from langchain_huggingface import HuggingFaceEmbeddings
# # Choose from https://huggingface.co/spaces/mteb/leaderboard

# # embedding = HuggingFaceEmbeddings(model_name="lier007/xiaobu-embedding-v2")

# model_path = os.path.join(os.environ['SENTENCE_TRANSFORMERS_HOME'], 'models--lier007--xiaobu-embedding-v2/snapshots/ee0b4ecdf5eb449e8240f2e3de2e10eeae877691')
# embedding = HuggingFaceEmbeddings(model_name=model_path)


# In[ ]:


from langchain_community.vectorstores import Neo4jVector

lc_retrieval_query = """
WITH collect(node) as nodes
// Entity - Text Unit Mapping
WITH
collect {
    UNWIND nodes as n
    MATCH (n)<-[:HAS_ENTITY]->(c:__Chunk__)<-[:HAS_CHILD]->(p:__Parent__)
    WITH c, p, count(distinct n) as freq
    RETURN {content: p.content, source: p.source} AS chunkText
    ORDER BY freq DESC
    LIMIT $topChunks
} AS text_mapping,
// Entity - Report Mapping
collect {
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
    WHERE c.summary is not null
    WITH c, c.rank as rank, c.weight AS weight
    RETURN c.summary 
    ORDER BY rank, weight DESC
    LIMIT $topCommunities
} AS report_mapping,
// Outside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r]-(m) 
    WHERE NOT m IN nodes and r.description is not null
    RETURN {description: r.description, sources: r.sources} AS descriptionText
    ORDER BY r.rank, r.weight DESC 
    LIMIT $topOutsideRels
} as outsideRels,
// Inside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r]-(m) 
    WHERE m IN nodes and r.description is not null
    RETURN {description: r.description, sources: r.sources} AS descriptionText
    ORDER BY r.rank, r.weight DESC 
    LIMIT $topInsideRels
} as insideRels,
// Entities description
collect {
    UNWIND nodes as n
    match (n)
    WHERE n.description is not null
    RETURN {description: n.description, sources: n.sources} AS descriptionText
} as entities
// We don't have covariates or claims here
RETURN {Chunks: text_mapping, Reports: report_mapping, 
       Relationships: outsideRels + insideRels, 
       Entities: entities} AS text, 1.0 AS score, {} AS metadata
"""

vectorstore = Neo4jVector.from_existing_graph(embedding=embedding, 
                                    index_name="embedding",
                                    node_label='__Entity__', 
                                    embedding_node_property='embedding', 
                                    text_node_properties=['id', 'description'],
                                    retrieval_query=lc_retrieval_query)
topChunks = 3
topCommunities = 3
topOutsideRels = 10
topInsideRels = 10
topEntities = 10
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "2wsx3edc"

local_search_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.9,
                   'k': topEntities,
                   'params': {
                        "topChunks": topChunks,
                        "topCommunities": topCommunities,
                        "topOutsideRels": topOutsideRels,
                        "topInsideRels": topInsideRels,
                    }},
    tags=['GraphRAG']
)
local_search_retriever


# In[25]:


# print(local_search_retriever.invoke('保險費暨保險單借款利息自動轉帳付款授權')[0].page_content)


# In[26]:


from tools.TWLF_Neo4jVector import TWLF_Neo4jVector
vectorstore = TWLF_Neo4jVector.from_existing_graph(
                                    embedding=embedding, 
                                    index_name="chunk_index",
                                    node_label='__Chunk__', 
                                    embedding_node_property='embedding', 
                                    text_node_properties=['content'])
vector_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.9},
    tags=['RAG']
)


# In[27]:


import os
# from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
llm


# In[9]:


# from langchain_ollama import ChatOllama
# llm = ChatOllama(
#     # model="llama3.1:70b-instruct-q8_0",
#     model='qwen2:72b-instruct-q8_0',
# )
# llm


# In[28]:


from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
"""
你是一個有用的助手, 你的任務是整理提供的資訊, 使用長度與格式符合「multiple paragraphs」針對使用者的問題來回應,
提供的資訊包含 檢索內容、圖譜資料庫相關節點與關係資訊, 無關的資訊直接忽略
你必須使用繁體中文回應問題, 盡可能在500字內回應,
若提供的資訊全部無關聯, 回應「找不到相關資訊」並結束, 不要捏造任何資訊,
最終的回應將清理後的訊息合併成一個全面的答案，針對回應的長度和格式對所有關鍵點和含義進行解釋
根據回應的長度和格式適當添加段落和評論。以Markdown格式撰寫回應。
回應應保留原有的意思和使用的情態動詞，例如「應該」、「可以」或「將」。
請確保使用繁體中文回答問題


以下為檢索內容:
"{context}"

以下為圖譜資料庫相關節點(Entities)、關係(Relationships)、社群(Reports)、Chunks(內文節錄)資訊:
"{graph_result}"

問題: {question}
"""
)

rag_chain = (
    {"context": vector_retriever, "question": RunnablePassthrough(), "graph_result": local_search_retriever}
    | prompt
    | llm
    | StrOutputParser()
)


# In[30]:


# import sys
# sys.path.append('..')
# from tools.TokenCounter import num_tokens_from_string

# q = '首期匯款帳號有哪些銀行, 匯款帳號跟劃撥帳號是什麼?'
# print(num_tokens_from_string(q))


# In[31]:


# for r in rag_chain.stream(q):
#     print(r, end="", flush=True)


# In[32]:


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question " # 給定一段聊天歷史和使用者的最新問題
    "which might reference context in the chat history, " # 這個問題可能會引用聊天歷史中的上下文
    "formulate a standalone question which can be understood " # 請將問題重新表述為一個獨立的問題，使其在沒有聊天歷史的情況下也能被理解
    "without the chat history. Do NOT answer the question, " # 不要回答這個問題
    "just reformulate it if needed and otherwise return it as is." # 只需在必要時重新表述問題，否則原樣返回
    "請確保使用繁體中文回應"
)

# 定義上下文解析的Chain
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

contextualize_chain = (
    contextualize_q_prompt
    | llm
    | StrOutputParser().with_config({
        'tags': ['contextualize_question']
    })
)


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel, RunnableLambda

store = {}

    
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 使用 RunnableParallel 來組織多個並行查詢
context_and_search_chain = RunnableParallel(
    {
        "context": RunnableLambda(lambda inputs: vector_retriever.invoke(inputs)),
        "graph_result": RunnableLambda(lambda inputs: local_search_retriever.invoke(inputs)),
        "question": lambda x: x,  # 保留原始輸入
    }
)

rag_chain = (
    contextualize_chain
    | context_and_search_chain
    | prompt
    | llm
    | StrOutputParser().with_config({
        "tags": ['final_output']
    })
)
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)


# In[35]:


# for r in conversational_rag_chain.stream(
#     {"question": "首期匯款帳號有哪些銀行, 帳號是什麼?"},
#     config={
#         "configurable": {"session_id": "abc1234"}
#     },  # constructs a key "abc123" in `store`.
# ):
#     print(r , end="")


# In[ ]:


from langchain_core.output_parsers.string import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes
from typing import List, Tuple
from pydantic import BaseModel, Field

class ChatHistory(BaseModel):
    """Chat history with the bot."""
    question: str
    
conversational_rag_chain = (
  conversational_rag_chain | StrOutputParser()
).with_types(input_type=ChatHistory)

# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

add_routes(
    app,
    conversational_rag_chain
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)


# In[ ]:





# In[ ]:




