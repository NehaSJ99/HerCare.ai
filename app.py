import streamlit as st
from google.genai import Client

# Initialize the Gemini client with the API key (hard-coded)
gemini_client = Client(api_key="AIzaSyDekHmFWhgsom8b_5YVJn_HR8kdXErOUMA")

def get_gemini_response(user_input):
    try:
        # Construct a prompt for the Gemini API
        prompt = f"Provide advice on managing the following situation: {user_input}"
        
        # Call Gemini API to generate content
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",  # Replace with the correct model if needed
            contents=prompt
        )
        
        # Access the response text directly (adjust attribute name if necessary)
        if hasattr(response, 'text'):
            return response.text
        else:
            # If the attribute is named differently, try 'generated_text'
            return getattr(response, 'generated_text', "No response generated.")
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.set_page_config(page_title="HerCare - Period Care Recommender", page_icon="👩‍⚕️")
    st.title("HerCare AI - Your Menstrual Health Assistant 💖")
    st.write("🤗 Ask about menstrual health, symptoms, care routines, and more!")
    
    user_input = st.text_area("Ask HerCare anything about your menstrual cycle:")
    
    if st.button("Get Advice"):
        if user_input:
            with st.spinner("Thinking..."):
                response = get_gemini_response(user_input)
                st.success("Here’s your advice:")
                st.write(response)
        else:
            st.warning("Please enter a question before clicking the button.")

if __name__ == "__main__":
    main()
