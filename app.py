from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

summarizer = None

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
    return {"status": "AI summarizer running"}


@app.post("/summarize")
def summarize(text: str):
    model = get_model()
    result = model(text, max_length=100, min_length=30)
    return {"summary": result[0]["summary_text"]}
