import sys
sys.path.append('..')
from tools.logger import getLogger
from tools.PDFTablePyPlumberLoader import PDFTablePyPlumberLoader
from flask import Flask, request, jsonify
from langchain_core.documents import Document
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
import os
import traceback
from langchain_core.prompts import ChatPromptTemplate
from flask_cors import CORS, cross_origin

logging = getLogger()

# 初始化 Flask 應用
app = Flask(__name__)
CORS(app)  # 允许所有域名的跨域请求

# 初始化 LLM
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0
)
class QuestionAnswer(BaseModel):
    question: str = Field(
        description="客戶提出的問題"
    )
    answer: str = Field(
        description="相對應的答案"
    )
    
class QuestionAnswerPack(BaseModel):
    question_answers: Optional[List[QuestionAnswer]] = Field(
        description="客戶可能提出的問題與答案清單"
    )
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
"""你是一個客戶服務問答專家，根據提供的資訊，提供客戶可能提的問題，以及相對自然語言回應的答案，
如果提供的資訊不足以產生問答，不要捏造任何未提供的資訊，你可以縮小問答數量來提供正確的資訊
"""
    ),
    (
        "human",
        """請根據參考內容，模擬客戶可能會問的問題與答案，提供 {question_num} 個問答\n\n提供的資訊:\n{ref_str}""",
    ),
])
desc_llm = llm.with_structured_output(
    QuestionAnswerPack
)
desc_chain = prompt | desc_llm

# 生成問答的函數
def qa_generator(question_num: int, ref_str: str):
    try:
        if question_num > 50:
            question_num = 50
        if question_num < 1:
            question_num = 1
        qas = desc_chain.invoke({"ref_str": ref_str, "question_num": question_num}).question_answers
        if not qas:
            return []
        return qas
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        return []

# API 1：接收檔案並產生完整檔案內容
@app.route('/generate_file_content', methods=['POST'])
def generate_file_content():
    file = request.files['file']
    save_dir = '../tmp'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)
    file.save(save_path)
    # 使用你提供的 PDF 處理邏輯
    pages: List[List[Document]] = []
    loader = PDFTablePyPlumberLoader(save_path, llm)
    pages = loader.load()
    # delete file
    os.remove(save_path)
    return jsonify([doc.dict() for doc in pages])

# API 2：接收檔案內容並產生 QA
@app.route('/generate_qa', methods=['POST'])
def generate_qa():
    data = request.json
    documents = data.get('documents', [])
    question_num = data.get('question_num', 10)
    
    # 生成 QA
    qas: List[QuestionAnswer] = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        future_map = {}
        for doc in documents:
            future = executor.submit(qa_generator, question_num, doc['page_content'])
            futures.append(future)
            future_map[future] = doc
        for future in as_completed(futures):
            try:
                qas += future.result()
            except Exception as e:
                logging.error(f"Error processing document: {e}")
                logging.error(traceback.format_exc())
    
    return jsonify([qa.model_dump() for qa in qas])

if __name__ == '__main__':
    app.run(debug=True)
