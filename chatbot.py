import os
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Set Google API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key is missing. Please provide it in the .env file.")

# Function to retrieve hardcoded text from PDF files
def get_document_text():
    doc1_path = "C:/Users/chall/OneDrive/Desktop/Assignment/google.pdf"
    doc2_path = "C:/Users/chall/OneDrive/Desktop/Assignment/uber.pdf"
    doc3_path = "C:/Users/chall/OneDrive/Desktop/Assignment/tesla.pdf"
    
    text = ""
    for doc_path in [doc1_path, doc2_path, doc3_path]:
        pdf_reader = PdfReader(doc_path)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to create the conversational chain for querying
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say "Answer is not available in the context." 
    Don't provide the wrong answer.

    Context: {context}

    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        temperature=0.3, 
        google_api_key=GOOGLE_API_KEY
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process the user's query
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GOOGLE_API_KEY
    )
    # Load the FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Search the FAISS index for the most relevant documents
    docs = new_db.similarity_search(user_question)
    
    # Get the conversational chain
    chain = get_conversational_chain()
    
    # Pass the documents and the question to the chain
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response["output_text"]

# Streamlit UI setup
st.set_page_config(page_title="Document Genie", layout="wide")

def main():
    st.header("AI Chatbot")
    
    # Retrieve and process hardcoded PDF text
    if st.button("Process Documents"):
        with st.spinner("Processing..."):
            raw_text = get_document_text()
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Documents have been processed and indexed!")

    # Ask a question
    user_question = st.text_input("Ask a Question from the uploaded PDF Files")
    
    if user_question:
        response = user_input(user_question)
        st.write(f"Answer: {response}")

if __name__ == "__main__":
    main()
