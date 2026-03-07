from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

summarizer = pipeline(
    "summarization",
    model="phoodis/thai-news-summarizer"
)

@app.get("/")
def home():
    return {"status": "AI summarizer running"}

@app.post("/summarize")
def summarize(text: str):
    result = summarizer(text, max_length=100, min_length=30)
    return {"summary": result[0]["summary_text"]}
