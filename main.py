# 📁 main.py
import json
from scripts.fetch_split_embed import fetch_text_from_url, split_text, create_vectorstore, save_vectorstore


# Load URLs
with open("data/urls.json", "r") as f:
    url_data = json.load(f)
    urls = url_data["urls"]

all_chunks = []

for url in urls:
    print(f"Processing: {url}")
    text = fetch_text_from_url(url)
    if text:
        chunks = split_text(text)
        all_chunks.extend(chunks)

# Embed & Store
db = create_vectorstore(all_chunks)
save_vectorstore(db)
print("Embeddings saved to FAISS!")
