import streamlit as st
from google.genai import Client
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load knowledge base
def load_knowledge_base():
    with open("knowledge_base.json", "r", encoding="utf-8") as file:
        return json.load(file)

knowledge_base = load_knowledge_base()

# Initialize the Gemini client
gemini_client = Client(api_key="YOUR_API_KEY")  # Replace with your API key

# Function to find the most relevant answer from knowledge base
def retrieve_answer(user_input):
    questions = [item["question"] for item in knowledge_base]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions + [user_input])
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_match_index = similarity_scores.argmax()
    
    if similarity_scores[0, best_match_index] > 0.3:  # Confidence threshold
        return knowledge_base[best_match_index]["answer"]
    return None

# Function to get AI response (with retrieval)
def get_gemini_response(user_input, history):
    try:
        retrieved_answer = retrieve_answer(user_input)
        if retrieved_answer:
            return retrieved_answer

        # Construct a context-aware prompt
        chat_history = "\n".join(history[-3:])  # Keep the last 3 interactions
        prompt = f"Previous chat history:\n{chat_history}\nUser: {user_input}\nAI:"
        
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        
        return response.text if hasattr(response, 'text') else "I'm not sure, but I can try to help!"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit chatbot UI
def main():
    st.set_page_config(page_title="HerCare AI - Menstrual Health Chatbot", page_icon="🌸", layout="wide")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("HerCare AI - Your Menstrual Health Chatbot 💖")
    st.write("🤗 Chat with me about menstrual health, PCOS, periods, menopause, fertility, and more!")

    # Chat message container
    chat_container = st.container()
    
    with chat_container:
        for chat in st.session_state.chat_history:
            role, message = chat
            if role == "user":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(f"**HerCare AI:** {message}")

    # User input
    user_input = st.text_input("Ask me anything about menstrual health:", key="user_input")
    
    if st.button("Send"):
        if user_input.strip():
            st.session_state.chat_history.append(("user", user_input))

            with st.spinner("Thinking..."):
                response = get_gemini_response(user_input, [msg for _, msg in st.session_state.chat_history])
            
            st.session_state.chat_history.append(("ai", response))
            st.experimental_rerun()

if __name__ == "__main__":
    main()
