from fastapi import FastAPI
from pydantic import BaseModel
import requests
import re
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv

# OpenAI
from openai import OpenAI

load_dotenv()

app = FastAPI(
    title="Text Summarizer App",
    description="Hybrid Summarization (HF + OpenAI)",
    version="2.0"
)

# 🔹 Config
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_TOKEN = os.getenv("HF_TOKEN")
USE_OPENAI_FALLBACK = os.getenv("USE_OPENAI_FALLBACK", "true").lower() == "true"

# 🔹 OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip()

# 🔥 Hugging Face call
def hf_summarize(text):
    try:
        response = requests.post(
            HF_API_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json={"inputs": text},
            timeout=60
        )

        print("HF RAW:", response.text)

        data = response.json()

        if isinstance(data, list):
            result = data[0]
            return result.get("summary_text") or result.get("generated_text")

        return None

    except Exception as e:
        print("HF ERROR:", e)
        return None

# 🔥 OpenAI fallback
def openai_summarize(text):
    try:
        prompt = f"Summarize the following text in 2-3 concise sentences:\n\n{text}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.5
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("OpenAI ERROR:", e)
        return None

# 🚀 API
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    cleaned = clean_data(dialogue_input.dialogue)

    # 1️⃣ Try Hugging Face
    summary = hf_summarize(cleaned)

    # 2️⃣ Fallback to OpenAI
    if not summary and USE_OPENAI_FALLBACK:
        print("Using OpenAI fallback...")
        summary = openai_summarize(cleaned)

    if not summary:
        return {"error": "Both HF and OpenAI failed"}

    return {"summary": summary}

# UI
@app.get("/")
async def home():
    return FileResponse(os.path.join("templates", "index.html"))