from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os


def build_qa_chain(vector_db):

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",  
        temperature=0.2                # 🔥 hallucination reduce
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant.
Answer ONLY from the given PDF context.

Rules:
- Do NOT repeat sentences
- Keep answer concise
- If asked "in 2 lines", answer ONLY in 2 lines
- If answer is not in PDF, say "Not found in the document"

Context:
{context}

Question:
{question}

Answer:
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
