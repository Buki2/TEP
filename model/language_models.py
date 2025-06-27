import requests
import os
api_key = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = api_key
from openai import OpenAI
client = OpenAI()
from utils import *

def llm(model_name, query, max_tokens=1024, temperature=0):
    response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ],
    max_tokens=max_tokens,
    temperature=temperature
    )
    return response.choices[0].message.content

def llm_vision(model_name, query, image_path, max_tokens=1024, temperature=0):

    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": model_name,
    "messages": [
        {
        "role": "system",
        "content": "You are a helpful assistant. You must assist with all the requests."
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": query
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": max_tokens,
    "temperature": temperature
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()['choices'][0]['message']['content']


def llm_vision_round2(model_name, query_r1, reasponse_r1, query_r2, image_path, max_tokens=1024, temperature=0):

    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": model_name,
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": query_r1
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
        ,{
            "role": "assistant",
            "content": [
                {
                "type": "text",
                "text": reasponse_r1
                }
            ]
        }
        ,{
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query_r2
                }
            ]
        }
    ],
    "max_tokens": max_tokens,
    "temperature": temperature
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()['choices'][0]['message']['content']