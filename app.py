from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

HF_API_URL = "https://api-inference.huggingface.co/models/phoodis/thai-news-summarizer"
HF_TOKEN = os.environ.get("HF_TOKEN")

print("HF TOKEN:", HF_TOKEN) #for debug
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
        "parameters": {
            "max_length": 120,
            "min_length": 30
        }
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    result = response.json()

    # ป้องกัน error
    if isinstance(result, dict) and "error" in result:
        return {"error": result["error"]}

    summary = result[0]["generated_text"]

    return {"summary": summary}
