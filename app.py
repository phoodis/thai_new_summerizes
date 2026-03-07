from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

summarizer = None

class TextRequest(BaseModel):
    text: str

def get_model():
    global summarizer
    if summarizer is None:
        summarizer = pipeline(
            "summarization",
            model="phoodis/thai-news-summarizer"
        )
    return summarizer


@app.get("/")
def home():
    return {"status": "Thai News AI Summarizer running"}


@app.post("/summarize")
def summarize(req: TextRequest):
    model = get_model()
    result = model(req.text, max_length=120, min_length=30)
    return {"summary": result[0]["summary_text"]}
