import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from prompts.prompt_template import hercare_prompt_template

load_dotenv()

# Set up LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Load FAISS vectorstore
def load_vectorstore(path="embeddings/faiss_index"):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# Build ConversationalRetrievalChain WITHOUT internal memory
# We'll pass memory externally in the Streamlit app
def build_qa_chain(memory):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=hercare_prompt_template,
        return_source_documents=True,
        output_key="answer"  # explicitly set to avoid error
    )
    return chain
