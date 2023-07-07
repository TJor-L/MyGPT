from django.test import TestCase
from app.chains.chat import chat
import json


def memory_chat_test():
    history = [
        {
            'type': 'human',
            'data': {
                'content': 'hi!'
            }
        },
        {
            'type': 'ai',
            'data': {
                'content': 'whats up?'
            }
        },
        {
            'type': 'human',
            'data': {
                'content': 'I am Dijkstra'
            }
        },
        {
            'type': 'ai',
            'data': {
                'content': 'Hello Dijkstra'
            }
        }
    ]
    output = chat(query="Good Morning!", model_name="OpenAI",
                  with_memory=True, history=history)
    print(json.dumps(output))


if __name__ == "__main__":
    memory_chat_test()  # 调试chat_with_memory的过程
