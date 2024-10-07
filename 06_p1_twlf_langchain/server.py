#!/usr/bin/env python
import os
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

from langchain_openai import AzureOpenAIEmbeddings
emb = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint='https://sales-chatbot-llm.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15',
    azure_deployment='text-embedding-3-small',
    openai_api_version='2023-05-15'
)

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('../data/台灣人壽金美鑫美元利率變動型終身壽險.pdf')
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "，", "。", "【"])

splits = text_splitter.split_documents(docs)

from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(documents=splits, embedding=emb)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.5}
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)


from langchain.chains.combine_documents import create_stuff_documents_chain
### Answer question ###
system_prompt = (
    "你是一個有用的助手, 你的任務是回答問題."
    "你必須根據以下提供的檢索內容進行問答問題."
    "如果檢索內容為空, 則回答 '沒有找到相關資訊'"
    "以 5 至 10 句話以內回應, 保持答案的簡潔"
    "以下為檢索內容:\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


from fastapi import FastAPI
from langserve import add_routes

# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from typing import Any, List, Optional, Union
# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.
class Input(BaseModel):
    input: str
    # The field extra defines a chat widget.
    # Please see documentation about widgets in the main README.
    # The widget is used in the playground.
    # Keep in mind that playground support for agents is not great at the moment.
    # To get a better experience, you'll need to customize the streaming output
    # for now.
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "input", "output": "output"}},
    )

class Metadata(BaseModel):
    page: int
    source: str

class ContextItem(BaseModel):
    id: Optional[str]
    metadata: Metadata
    page_content: str
    type: str

class Output(BaseModel):
    context: Optional[List[ContextItem]]
    answer: Optional[str]
    input: Optional[str]
    

    
add_routes(
    app,
    conversational_rag_chain.with_types(input_type=Input, output_type=Output),
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
# Applciaition Demo URI: http://localhost:8000/chain/playground/

