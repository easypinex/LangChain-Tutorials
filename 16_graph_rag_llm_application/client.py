# ! pip install sseclient-py

QUESTION = '首期繳費有哪些方式?'

import sseclient
import uuid
input_json = {
    "input": {
        "history": [],
        "input": QUESTION
    },
    "config": {
        "configurable": {
            "session_id": str(uuid.uuid4())
        }
    }
}

import requests
import json
from copy import deepcopy

url = 'http://localhost:8000/stream_events'
stream_response = requests.post(url, json=input_json, stream=True)

client = sseclient.SSEClient(stream_response)
for event in client.events():
    data = json.loads(event.data)
    name = data.get('name')
    event = data.get('event')
    tags = data['tags']
    if (event == 'on_retriever_end' and name == 'Retriever'):
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
    elif 'final_output' in tags:
        chunk = data['data'].get('chunk')
        if chunk is not None:
            print(chunk, end="")
print("")
print("-" * 40)