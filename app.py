import streamlit as st
from google.genai import Client
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from knowledge_base import knowledge_base  # Import the knowledge base

# Initialize the Gemini client with API key
gemini_client = Client(api_key="AIzaSyDekHmFWhgsom8b_5YVJn_HR8kdXErOUMA")

# Function to retrieve the most relevant answer based on user input
def retrieve_relevant_answer(user_input):
    questions = [entry['question'] for entry in knowledge_base]
    answers = [entry['answer'] for entry in knowledge_base]
    
    # Use TF-IDF Vectorizer to convert text into numerical vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions + [user_input])
    
    # Calculate cosine similarity between user input and knowledge base questions
    cosine_similarities = np.dot(tfidf_matrix[-1], tfidf_matrix[:-1].T).toarray()
    
    # Get the index of the most similar question
    most_similar_idx = np.argmax(cosine_similarities)
    
    return answers[most_similar_idx]

# Function to call Gemini API with retrieved information
def get_gemini_response(user_input, retrieved_info):
    try:
        prompt = f"Provide advice on managing the following situation: {user_input}. Additional info: {retrieved_info}"
        
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        
        if hasattr(response, 'text'):
            return response.text
        else:
            return getattr(response, 'generated_text', "No response generated.")
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit app
def main():
    st.set_page_config(page_title="HerCare - Women's Health Assistant", page_icon="👩‍⚕️")
    st.title("HerCare AI - Your Women's Health Guide 💖")
    st.write("🌸 Ask about periods, ovulation, PCOS, pregnancy, and more!")

    user_input = st.text_area("Ask HerCare anything:")

    if st.button("Get Advice"):
        if user_input:
            with st.spinner("Thinking..."):
                retrieved_info = retrieve_relevant_answer(user_input)
                response = get_gemini_response(user_input, retrieved_info)
                
                st.success("Here’s your advice:")
                st.write(response)
        else:
            st.warning("Please enter a question before clicking the button.")

if __name__ == "__main__":
    main()
