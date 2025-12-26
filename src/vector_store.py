from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100000,
        chunk_overlap=20000 
    )

    chunks = splitter.split_documents(documents)

    # ✅ LOCAL EMBEDDINGS (NO GEMINI, NO API, NO QUOTA)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db
