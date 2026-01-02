import os
from pypdf import PdfReader
from langchain.schema import Document


def load_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)
    documents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": i + 1
                    }
                )
            )
    return documents
