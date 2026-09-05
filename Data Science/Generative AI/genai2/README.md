# 📚 RAG Pipeline & ChromaDB Vector Store (genai2)

A Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB vector storage, and document loaders.

## 🚀 Components
- **Document Loaders**: Load external text documents into structured data chunks.
- **Vector Store & Retrievers**: Store embeddings using ChromaDB and query relevant contexts.
- **`app.py` / `main.py`**: Query engine interface returning contextualized answers.

## 🛠 Setup & Running
1. Copy `.env.example` to `.env` and insert your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
2. Install dependencies and run:
   ```bash
   pip install -r requirements.txt
   python create_database.py
   python app.py
   ```
