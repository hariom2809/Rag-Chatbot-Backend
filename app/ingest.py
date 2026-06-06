import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.config import DATA_DIR, EMBEDDING_MODEL, VECTORSTORE_DIR
else:
    from .config import DATA_DIR, EMBEDDING_MODEL, VECTORSTORE_DIR


def ingest_pdfs():
    print("Loading PDFs from data/ folder...")

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # Load all PDFs in the data directory
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()

    if not documents:
        print("No PDF pages were found to ingest.")
        return

    print(f"Loaded {len(documents)} pages from PDFs")

    # Split into chunks
    # chunk_size = how many characters per chunk
    # chunk_overlap = overlap between adjacent chunks (avoids losing context at boundaries)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Load embedding model
    print("Loading embedding model (downloads on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    # Create FAISS index from all chunks
    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save to disk so RAG chain can load it later
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"✅ Vector store saved to {VECTORSTORE_DIR}")


if __name__ == "__main__":
    ingest_pdfs()