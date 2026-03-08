from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

HF_API_URL = "https://router.huggingface.co/hf-inference/models/phoodis/thai-news-summarizer"
HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"status": "Thai News AI Summarizer running"}


@app.post("/summarize")
def summarize(req: TextRequest):

    payload = {
        "inputs": req.text,
        "options": {"wait_for_model": True},
        "parameters": {
            "max_length": 120,
            "min_length": 30
        }
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    result = response.json()

    print("HF RESPONSE:", result)

    if isinstance(result, list):
        summary = result[0].get("generated_text", "")
        return {"summary": summary}

    return {"error": result}
