import sys
import os

# 🔑 IMPORTANT: project root path add (for src imports)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv

from src.pdf_loader import load_pdf
from src.vector_store import create_vector_store
from src.qa_chain import build_qa_chain

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(
    page_title="PDF RAG Chatbot (Gemini)",
    page_icon="📘",
    layout="centered"
)

st.title("📘 PDF RAG Chatbot using Gemini")

# 📂 PDF path (CHANGE HERE if needed)
PDF_PATH = r"D:\ngrok-v3-stable-windows-amd64\intern\data\Ebook -Computer Organisation and Design (2014).pdf"

@st.cache_resource(show_spinner=True)
def setup_rag_pipeline():
    """
    1. Load PDF
    2. Chunk + Embed
    3. Build FAISS vector DB
    4. Create RAG QA chain
    """
    docs = load_pdf(PDF_PATH)
    vector_db = create_vector_store(docs)
    qa_chain = build_qa_chain(vector_db)
    return qa_chain

# Initialize RAG
qa_chain = setup_rag_pipeline()

# User input
query = st.text_input("Ask a question from the PDF:")

if query:
    with st.spinner("Thinking... 🤔"):
        answer = qa_chain.run(query)

    st.subheader("Answer")
    st.write(answer)
