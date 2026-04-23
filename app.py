from fastapi import FastAPI
from pydantic import BaseModel
import requests
import re
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv
load_dotenv()


app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

#  Hugging Face API (your trained model)
API_URL = "https://api-inference.huggingface.co/models/abhijeetrai01/text-summarizer"

#  IMPORTANT: apna token yaha daalo


headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

# request model
class DialogueInput(BaseModel):
    dialogue: str

# clean function
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()

# API call function
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# summarization endpoint
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    cleaned = clean_data(dialogue_input.dialogue)

    output = query({
        "inputs": cleaned,
        "parameters": {"max_length": 150}
    })

    try:
        summary = output[0]["summary_text"]
    except:
        summary = str(output)

    return {"summary": summary}

# UI route
@app.get("/")
async def home():
    return FileResponse(os.path.join("templates", "index.html"))