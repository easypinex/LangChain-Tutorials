'''
示例如何透過 LangServe, 提供之API stream_events, 獲取RAG資訊以及回應內容
'''

import uuid
import json
from copy import deepcopy

import sseclient
import requests


# ! pip install sseclient-py

QUESTION = '首期繳費有哪些方式?'
URL = 'http://localhost:8000/stream_events'

input_json = {
    "input": {
        "chat_history": [],
        "question": QUESTION
    },
    "config": {
        "configurable": {
            "session_id": str(uuid.uuid4())
        }
    }
}

stream_response = requests.post(URL, json=input_json, stream=True, timeout=15)

client = sseclient.SSEClient(stream_response)
for event in client.events():
    data = json.loads(event.data)
    name = data.get('name')
    event = data.get('event')
    tags = data['tags']
    if event == 'on_retriever_end' and name == 'Retriever':
        isGraphRAG = 'GraphRAG' in tags
        documents = data['data']['output']['documents']
        for document in documents:
            print('來源: ' + ('【Graph】' if isGraphRAG else document['metadata']['source']))
            content = document['page_content'].strip()
            if content.startswith('content:'):
                content = content[len('content:'):]
            content = content.strip()
            metadata = deepcopy(document['metadata'])
            if 'source' in metadata:
                del metadata['source']
            print('內文: ' + content)
            print('參數: ' + str(metadata))
            print('-' * 40)
    elif event == 'on_parser_end' and 'contextualize_question' in tags:
        print('問題更新: ' + data['data']['output'])
        print('-' * 40)
    elif 'final_output' in tags:
        chunk = data['data'].get('chunk')
        if chunk is not None:
            print(chunk, end="")
print("")
print("-" * 40)
