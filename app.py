from fastapi import FastAPI
from pydantic import BaseModel
import requests
import re
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)

#  Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/abhijeetrai01/text-summarizer"

headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()

#  FINAL SAFE QUERY FUNCTION
def query(payload):
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        #  DEBUG RAW RESPONSE
        print("RAW RESPONSE:", response.text)

        #  Empty response check
        if not response.text:
            return {"error": "Empty response from Hugging Face API"}

        #  Safe JSON parse
        try:
            return response.json()
        except:
            return {"error": response.text}

    except Exception as e:
        return {"error": str(e)}

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    #  T5 FIX
    cleaned = "summarize: " + clean_data(dialogue_input.dialogue)

    output = query({
        "inputs": cleaned,
        "parameters": {
            "max_length": 150,
            "min_length": 30
        },
        "options": {
            "wait_for_model": True
        }
    })

    print("HF OUTPUT:", output)

    #  Handle error
    if isinstance(output, dict):
        return {"error": output}

    try:
        result = output[0]

        summary = result.get("summary_text") or result.get("generated_text")

        if not summary:
            return {
                "error": "No summary returned",
                "raw_output": output
            }

    except Exception:
        return {"error": str(output)}

    return {"summary": summary}

@app.get("/")
async def home():
    return FileResponse(os.path.join("templates", "index.html"))