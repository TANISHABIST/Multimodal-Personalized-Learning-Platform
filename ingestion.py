"""
Data ingestion: load PDFs and split into chunks.
"""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_single_pdf(pdf_path: str):
    """Load a single PDF and tag it with source metadata."""
    pdf_path = Path(pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_file"] = pdf_path.name
        doc.metadata["file_type"] = "pdf"

    return documents


def process_all_pdfs(pdf_directory):
    
    all_documents = []
    pdf_dir = Path(pdf_directory)

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to Process")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            documents = process_single_pdf(pdf_file)
            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")
        except Exception as e:
            print(f"Error: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200, min_chunk_len=50):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = text_splitter.split_documents(documents)

    split_docs = [
        chunk for chunk in split_docs
        if len(chunk.page_content.strip()) >= min_chunk_len
    ]

    print(f"Split {len(documents)} document into {len(split_docs)} chunks")

    return split_docs