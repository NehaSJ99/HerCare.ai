import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def fetch_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove scripts, styles, etc.
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

load_dotenv()
embeddings = OpenAIEmbeddings()

def create_vectorstore(chunks):
    return FAISS.from_texts(chunks, embeddings)

def save_vectorstore(db, path="embeddings/faiss_index"):
    db.save_local(path)

