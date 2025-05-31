import streamlit as st
from langchain.memory import ConversationBufferMemory
from scripts.querychain import build_qa_chain

st.set_page_config(page_title="HerCare.ai Chat", layout="wide")
st.title("HerCare.ai – Women's Health Assistant")

# Setup memory in session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# Setup chain in session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = build_qa_chain(memory=st.session_state.memory)

# Store full chat history for UI rendering
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_question = st.text_input("Ask about PCOS, menstruation, menopause...")

if user_question:
    # Store user question in UI history
    st.session_state.chat_history.append(("user", user_question))

    with st.spinner("Thinking..."):
        result = st.session_state.qa_chain.invoke({"question": user_question})
        answer = result["answer"]

    # Store assistant answer in UI history
    st.session_state.chat_history.append(("ai", answer))

# Display full chat
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**HerCare:** {msg}")
