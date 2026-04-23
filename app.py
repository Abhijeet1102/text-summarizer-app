from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.responses import FileResponse
import os

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# load model


model = T5ForConditionalGeneration.from_pretrained("abhijeetrai01/text-summarizer")
tokenizer = T5Tokenizer.from_pretrained("abhijeetrai01/text-summarizer")

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# request model
class DialogueInput(BaseModel):
    dialogue: str

# clean function
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()

# summarization function
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

# API
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

# UI route (FIXED)
@app.get("/")
async def home():
    return FileResponse(os.path.join("templates", "index.html"))