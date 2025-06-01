# HerCare AI Chatbot 2.0

HerCare AI Chatbot is an AI-powered assistant designed to provide reliable, personalized support for menstrual health. Version 2.0 introduces a modular, production-ready backend using LangChain's LangServe, FAISS-powered RAG (Retrieval-Augmented Generation), and conversational memory capabilities. This release is optimized for API-based deployment and seamless integration into both research and healthcare applications.

## What's New in v2.0

- Conversational memory support for contextual multi-turn interactions
- LangServe-based FastAPI deployment for scalable API hosting
- FAISS-powered RAG system using curated medical knowledge sources
- Streamlit and FastAPI support for dual interaction modes
- Containerized deployment with Hugging Face Spaces (Docker-based)

## Legacy v1.x Overview

The initial version of HerCare AI Chatbot was a Streamlit-based interface providing cycle tracking insights, symptom-based wellness tips, and educational content using Gemini AI with basic RAG. It focused on user experience and personalized interaction in a simple web interface.

### Key Features from v1.x

- AI-powered assistance for menstrual and reproductive health
- Symptom-driven guidance using RAG pipelines
- Cycle prediction support and self-care education
- Lightweight Streamlit-based interface

## Architecture Overview

- **LLM**: Gemini AI or OpenAI models via LangChain interface
- **RAG**: FAISS vector store built from medical articles
- **Memory**: ConversationBufferMemory for chat context retention
- **Frontend**: Streamlit (for interactive mode)
- **Backend**: LangServe (FastAPI API for integration/deployment)
- **Deployment**: Hugging Face Spaces (Docker)

## Getting Started

### Streamlit Chat Interface (for testing)

```bash
git clone https://github.com/your-username/HerCare-AI-Chatbot.git
cd HerCare-AI-Chatbot
pip install -r requirements.txt
streamlit run test.py
```

### API Deployment (Local)

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```
Access the OpenAPI docs at:
```
http://localhost:8000/hercare/docs
```

### Deploy to Hugging Face Spaces

1. Create a new Hugging Face Space with the **Docker** SDK option.
2. Add the following files to the root of the repository:
   - `serve.py`
   - `Dockerfile`
   - `requirements.txt`
3. Push the code to the remote space.

Hugging Face will build the image and deploy your FastAPI app at:
```
https://<username>-<space-name>.hf.space/hercare/docs
```

## Requirements

- Python 3.10+
- OpenAI API Key (for embedding + inference)
- `faiss-cpu`, `langserve[all]`, `uvicorn`, `python-dotenv`

## Disclaimer

HerCare AI is an educational tool. It does not provide medical diagnoses or treatment recommendations. Users should consult licensed healthcare providers for any personal medical concerns.
