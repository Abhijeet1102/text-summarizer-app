#  Text Summarizer App (FastAPI + AI)

A simple and powerful Text Summarization web application built using **FastAPI** with a **hybrid AI approach**.

This app summarizes long text into concise summaries using:

*  Hugging Face (BART model)
*  OpenAI (fallback for reliability)

---

##  Live Demo

-> https://text-summarizer-app-74op.onrender.com

---

##  Features

*  Summarize long text instantly
*  Hybrid AI system (Hugging Face + OpenAI)
*  Automatic fallback if one service fails
*  Clean and simple UI
*  FastAPI backend (high performance)
*  Deployed on Render

---

## ⚙️ Tech Stack

* **Backend:** FastAPI (Python)
* **AI Models:**

  * Hugging Face → facebook/bart-large-cnn
  * OpenAI → gpt-4o-mini
* **Frontend:** HTML (Jinja templates)
* **Deployment:** Render
* **Libraries:** requests, python-dotenv

---

##  How It Works

User Input
↓
Hugging Face API (BART model)
↓
If fails → OpenAI fallback
↓
Final Summary

---

##  Installation (Local Setup)

git clone https://github.com/Abhijeet1102/text-summarizer-app.git
cd text-summarizer-app

---

##  Create Virtual Environment

python -m venv venv
venv\Scripts\activate

---

##  Install Dependencies

pip install -r requirements.txt

---

##  Environment Variables

Create a `.env` file in the root folder:

OPENAI_API_KEY=your_openai_key
HF_TOKEN=your_huggingface_token
USE_OPENAI_FALLBACK=true

---

##  Run Locally

uvicorn app:app --reload

-> Open in browser:
http://127.0.0.1:8000

---

## 📡 API Endpoint

### POST /summarize/

### Request:

{
"dialogue": "Artificial Intelligence is transforming industries..."
}

### Response:

{
"summary": "AI is transforming industries but raises ethical concerns."
}

---

##  Why Hybrid Approach?

* Hugging Face → Fast & free but sometimes unstable
* OpenAI → Reliable but paid

-> Combining both ensures:

*  High availability
*  Better user experience
*  No downtime

---

##  Model Usage & Future Scope

###  Current Models Used

*  Hugging Face → facebook/bart-large-cnn
*  OpenAI → gpt-4o-mini (fallback)

---

###  Custom Trained Model (Important)

A custom summarization model has already been:

*  Trained using T5 architecture
*  Uploaded on Hugging Face
*  Model: abhijeetrai01/text-summarizer

---

###  Why Custom Model is NOT used currently

* Render free plan provides only **512 MB RAM**
* The trained model size is **500MB+**
* Not enough memory to load model directly
* Hugging Face free inference API for custom models is:

  *  Slow (cold start)
  *  Unstable (timeouts)

---

###  Future Plan

The custom model will be used by:

* Deploying via **Hugging Face Inference Endpoints**
* Connecting endpoint with FastAPI backend
* Replacing BART with custom trained model

---

###  Cost Consideration

* Hugging Face endpoints are **paid services**
* Currently not used to keep project **free**
* Can be enabled anytime by adding credits

---

###  Engineering Decision

This project demonstrates a real-world trade-off:

* Present → Stable & free (BART + OpenAI)
* Future → Custom & scalable (T5 + Endpoint)

---

##  Author

**Abhijeet Rai**
MCA Student
Interested in AI & Backend Development

---

##  Future Improvements

* UI enhancements (loader, copy button)
* Multiple summary lengths (short/medium/long)
* User authentication
* Database integration

---

##  License

This project is open-source .
