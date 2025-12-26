from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def build_qa_chain(vector_db):
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
        convert_system_message_to_human=True  # 🔥 IMPORTANT FIX
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff"
    )
    return qa_chain
