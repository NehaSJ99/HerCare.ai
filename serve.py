from fastapi import FastAPI
from langserve import add_routes
from scripts.querychain import build_qa_chain
from langchain.memory import ConversationBufferMemory

app = FastAPI(title="HerCare.ai RAG API")

# Memory setup
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Build chain
qa_chain = build_qa_chain(memory=memory)

# Serve with LangServe
add_routes(
    app,
    qa_chain,
    path="/hercare"
)

