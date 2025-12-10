#!/usr/bin/env python3
"""
Index knowledge base documents (PDFs, logbooks, books) using RAG/vector embeddings

This script processes all documents in knowledge_base/{papers,logbooks,books}/
and creates a searchable vector database using ChromaDB.

Requirements:
    pip install chromadb sentence-transformers PyPDF2 python-docx

Usage:
    python knowledge_base/index_knowledge.py

The indexed database will be stored in knowledge_base/chroma_db/ and can be
queried instantly by the MCP server tools.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("Warning: chromadb not installed. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("Warning: PyPDF2 not installed. Install with: pip install PyPDF2")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file"""
    if not HAS_PDF:
        return ""

    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"  ⚠️  Error reading {pdf_path.name}: {e}")
    return text


def extract_text_from_txt(txt_path: Path) -> str:
    """Extract text from text file"""
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"  ⚠️  Error reading {txt_path.name}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for better context preservation"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def get_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of file for change detection"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def index_documents(kb_path: Path, collection_name: str = "hedm_knowledge"):
    """Index all documents in knowledge base"""
    if not HAS_CHROMA or not HAS_TRANSFORMERS:
        print("❌ Missing dependencies. Install with:")
        print("   pip install chromadb sentence-transformers PyPDF2")
        return

    print("🚀 Starting knowledge base indexing...")
    print()

    # Initialize embedding model (lightweight, runs locally)
    print("📦 Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB model, very fast
    print("   ✓ Model loaded")
    print()

    # Initialize ChromaDB
    chroma_path = kb_path / "chroma_db"
    chroma_path.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))

    # Delete existing collection if it exists (for fresh indexing)
    try:
        client.delete_collection(name=collection_name)
        print(f"   Deleted existing collection '{collection_name}'")
    except:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "HEDM knowledge base with papers, logbooks, and books"}
    )

    # Track indexing stats
    stats = {
        "papers": 0,
        "logbooks": 0,
        "books": 0,
        "total_chunks": 0
    }

    # Index papers
    papers_dir = kb_path / "papers"
    if papers_dir.exists():
        print("📄 Indexing papers...")
        for pdf_file in papers_dir.glob("*.pdf"):
            print(f"   Processing: {pdf_file.name}")
            text = extract_text_from_pdf(pdf_file)
            chunks = chunk_text(text, chunk_size=500)

            for i, chunk in enumerate(chunks):
                embedding = embedder.encode(chunk).tolist()
                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": str(pdf_file.name),
                        "type": "paper",
                        "chunk_index": i,
                        "file_hash": get_file_hash(pdf_file)
                    }],
                    ids=[f"{pdf_file.stem}_chunk_{i}"]
                )

            stats["papers"] += 1
            stats["total_chunks"] += len(chunks)
            print(f"      ✓ Indexed {len(chunks)} chunks")

    # Index logbooks
    logbooks_dir = kb_path / "logbooks"
    if logbooks_dir.exists():
        print()
        print("📓 Indexing logbooks...")
        for txt_file in list(logbooks_dir.glob("*.txt")) + list(logbooks_dir.glob("*.md")):
            print(f"   Processing: {txt_file.name}")
            text = extract_text_from_txt(txt_file)
            chunks = chunk_text(text, chunk_size=300)  # Smaller chunks for logbooks

            for i, chunk in enumerate(chunks):
                embedding = embedder.encode(chunk).tolist()
                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": str(txt_file.name),
                        "type": "logbook",
                        "chunk_index": i,
                        "file_hash": get_file_hash(txt_file)
                    }],
                    ids=[f"{txt_file.stem}_chunk_{i}"]
                )

            stats["logbooks"] += 1
            stats["total_chunks"] += len(chunks)
            print(f"      ✓ Indexed {len(chunks)} chunks")

    # Index books
    books_dir = kb_path / "books"
    if books_dir.exists():
        print()
        print("📚 Indexing books...")
        for book_file in books_dir.glob("*.pdf"):
            print(f"   Processing: {book_file.name}")
            text = extract_text_from_pdf(book_file)
            chunks = chunk_text(text, chunk_size=700)  # Larger chunks for books

            for i, chunk in enumerate(chunks):
                embedding = embedder.encode(chunk).tolist()
                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": str(book_file.name),
                        "type": "book",
                        "chunk_index": i,
                        "file_hash": get_file_hash(book_file)
                    }],
                    ids=[f"{book_file.stem}_chunk_{i}"]
                )

            stats["books"] += 1
            stats["total_chunks"] += len(chunks)
            print(f"      ✓ Indexed {len(chunks)} chunks")

    # Save indexing stats
    stats_file = kb_path / "data" / "index_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            **stats,
            "collection_name": collection_name,
            "embedding_model": "all-MiniLM-L6-v2",
            "last_indexed": "auto-generated"
        }, f, indent=2)

    print()
    print("=" * 60)
    print("✅ Indexing complete!")
    print(f"   Papers: {stats['papers']}")
    print(f"   Logbooks: {stats['logbooks']}")
    print(f"   Books: {stats['books']}")
    print(f"   Total chunks indexed: {stats['total_chunks']}")
    print(f"   Database location: {chroma_path}")
    print("=" * 60)
    print()
    print("🎯 Knowledge base is ready for querying via MCP server!")


def test_query(kb_path: Path, query: str = "What is Bragg's law?"):
    """Test the indexed knowledge base with a sample query"""
    if not HAS_CHROMA or not HAS_TRANSFORMERS:
        return

    print()
    print(f"🔍 Testing query: '{query}'")
    print()

    chroma_path = kb_path / "chroma_db"
    if not chroma_path.exists():
        print("❌ No indexed database found. Run indexing first.")
        return

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path=str(chroma_path))

    try:
        collection = client.get_collection(name="hedm_knowledge")
        query_embedding = embedder.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        print("Top 3 results:")
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            print(f"\n{i}. Source: {meta['source']} ({meta['type']})")
            print(f"   Relevance: {1 - dist:.2%}")
            print(f"   Excerpt: {doc[:200]}...")

    except Exception as e:
        print(f"❌ Error querying: {e}")


def main():
    """Main indexing function"""
    kb_path = Path(__file__).parent

    print("=" * 60)
    print("HEDM Knowledge Base Indexer")
    print("=" * 60)
    print()

    # Check if there are documents to index
    papers_count = len(list((kb_path / "papers").glob("*.pdf"))) if (kb_path / "papers").exists() else 0
    logbooks_count = len(list((kb_path / "logbooks").glob("*.txt")) + list((kb_path / "logbooks").glob("*.md"))) if (kb_path / "logbooks").exists() else 0
    books_count = len(list((kb_path / "books").glob("*.pdf"))) if (kb_path / "books").exists() else 0

    total_docs = papers_count + logbooks_count + books_count

    if total_docs == 0:
        print("⚠️  No documents found to index!")
        print()
        print("Add documents to:")
        print(f"   • Papers (PDFs): {kb_path / 'papers'}/")
        print(f"   • Logbooks (TXT/MD): {kb_path / 'logbooks'}/")
        print(f"   • Books (PDFs): {kb_path / 'books'}/")
        print()
        print("Then run this script again.")
        return

    print(f"Found {total_docs} documents:")
    print(f"   • {papers_count} papers")
    print(f"   • {logbooks_count} logbooks")
    print(f"   • {books_count} books")
    print()

    # Run indexing
    index_documents(kb_path)

    # Test with sample query
    test_query(kb_path, "What is a good calibration strain value?")


if __name__ == "__main__":
    main()
