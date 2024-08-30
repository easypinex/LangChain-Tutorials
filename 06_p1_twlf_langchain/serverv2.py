from operator import itemgetter
from typing import List, Tuple

from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_chroma import Chroma

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

import os
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_TEMPLATE = """
根據聊天記錄和最新的使用者問題(可能引用了聊天記錄中的上下文),
重新構建一個不依賴聊天記錄也能被理解的獨立問題.
如果問題不需要重新構建, 就直接返回使用者問題, 不要回答該問題.

聊天紀錄:
{chat_history}
使用者問題: {question}
獨立問題:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """
"你是一個有用的助手, 你的任務是回答問題."
"你必須根據以下提供的檢索內容進行問答問題."
"如果檢索內容為空, 則回答 '沒有找到相關資訊'"
"以 5 至 10 句話以內回應, 保持答案的簡潔"
"以下為檢索內容:\n\n"
"{context}"

問題: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer
# ----------------------------------------
# from langchain_ollama import ChatOllama
# model_name = "llama3.1"
# model = ChatOllama(model=model_name)

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
# ----------------------------------------

# from langchain_ollama import OllamaEmbeddings
# emb = OllamaEmbeddings(
#     model=model_name,
# )

emb = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    azure_endpoint='https://sales-chatbot-llm.openai.azure.com/openai/deployments/embedding-ada-002/embeddings?api-version=2023-05-15',
    azure_deployment='text-embedding-ada-002',
    openai_api_version='2023-05-15'
)
# ----------------------------------------

dir = '../data'
for filename in os.listdir(dir):
    docs = []
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(dir, filename))
        docs += loader.load()
    print(docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "，", "。", "【"])
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=emb)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.8}
)

_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | model
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str


conversational_qa_chain = (
    _inputs | _context | ANSWER_PROMPT | model | StrOutputParser()
)
chain = conversational_qa_chain.with_types(input_type=ChatHistory)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain, enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)