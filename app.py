from fastapi import FastAPI
from pydantic import BaseModel
import requests
import re
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using BART",
    version="1.0"
)

#  CORRECT Hugging Face API (WORKING)
API_URL = "https://api-inference.huggingface.co/pipeline/summarization/facebook/bart-large-cnn"

headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

# Request model
class DialogueInput(BaseModel):
    dialogue: str

# Clean function
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip()

#  SAFE API CALL
def query(payload):
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        print("RAW RESPONSE:", response.text)

        if not response.text:
            return {"error": "Empty response from Hugging Face"}

        try:
            return response.json()
        except:
            return {"error": response.text}

    except Exception as e:
        return {"error": str(e)}

#  SUMMARIZATION API
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    # ❗ BART ke liye NO prefix
    cleaned = clean_data(dialogue_input.dialogue)

    output = query({
        "inputs": cleaned
    })

    print("HF OUTPUT:", output)

    # Handle error
    if isinstance(output, dict):
        return {"error": output}

    try:
        result = output[0]

        # Handle both possible keys
        summary = result.get("summary_text") or result.get("generated_text")

        if not summary:
            return {
                "error": "No summary returned",
                "raw_output": output
            }

    except Exception:
        return {"error": str(output)}

    return {"summary": summary}

# UI route
@app.get("/")
async def home():
    return FileResponse(os.path.join("templates", "index.html"))