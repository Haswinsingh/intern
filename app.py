import os
import sys
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pdf_loader import load_pdf
from src.vector_store import create_vector_store
from src.qa_chain import build_qa_chain

load_dotenv()

st.set_page_config(page_title="Groq RAG PDF Chatbot", layout="wide")
st.title("Technovers AI 2.O📡")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔍 Processing PDF..."):
        docs = load_pdf("temp.pdf")
        vector_db = create_vector_store(docs)
        qa_chain = build_qa_chain(vector_db)

    st.success("✅ PDF processed successfully!")

    query = st.text_input("Ask a question from the PDF:")

    if query:
        with st.spinner("🤖 Generating answer..."):
            result = qa_chain.invoke({"query": query})

        st.subheader("Answer")
        st.write(result["result"])

        with st.expander("📄 Source Pages"):
            for doc in result["source_documents"]:
                st.write(
                    f"📌 Page {doc.metadata['page']} – {doc.metadata['source']}"
                )
