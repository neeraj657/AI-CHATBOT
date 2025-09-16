# AI-CHATBOT

An AI-powered chatbot that answers questions based on the content of uploaded PDF files using Google Generative AI and LangChain.

# Features
--Upload PDF Files: Upload multiple PDFs for processing.

--Intelligent Question Answering: The chatbot uses Google Generative AI to answer questions based on the provided document content.

--Contextual Responses: If the answer is not in the document, the chatbot will let you know.

# Prerequisites
-To use this project, you'll need:

  --Python 3.7 or higher

  --Google Generative AI API Key for accessing Google’s AI services

# Installation
1.Clone the Repository

bash

git clone https://github.com/your-username/your-repository-name.git

cd your-repository-name

2.Install Dependencies

Install required packages using:

bash

pip install -r requirements.txt

3.Set Up Environment Variables

Create a .env file in the project root and add your Google API Key:

makefile

GOOGLE_API_KEY=your_google_api_key_here

# Usage
1.Start the Streamlit App

Run the Streamlit app using:

bash

streamlit run chatbot.py

2.Upload PDFs and Ask Questions

  Upload one or more PDF files.

  Ask questions in the input box to get responses based on the content of the uploaded documents.

# File Structure
--chatbot.py: The main code for the chatbot, including Streamlit setup, PDF processing, and question-answering logic.

--requirements.txt: Lists all required dependencies.

--.env: Stores your Google API key. (Make sure this file is added to .gitignore to keep it private.)

# Dependencies
The key libraries and packages used in this project are:

Streamlit: For creating the web application interface.

PyPDF2: For reading and extracting text from PDF files.

FAISS: For efficient similarity searching and vector indexing.

LangChain: For text processing and creating conversational pipelines.

python-dotenv: For handling environment variables.

google-generativeai: For Google’s Generative AI access.

langchain-google-genai: For integrating Google Generative AI with LangChain.

# Example Workflow
Upload PDF Files: Upload PDF documents through the interface.

Document Processing: The PDFs are processed and chunked into text pieces.

Question Answering: Input a question based on the document content. The chatbot uses Google Generative AI and FAISS to find the most relevant answer.

# Troubleshooting
Google API Key Error: If you see an error about the Google API key, check that your .env file is set up correctly.

FAISS Compatibility: Ensure that faiss-cpu is compatible with your Python version and system architecture.
