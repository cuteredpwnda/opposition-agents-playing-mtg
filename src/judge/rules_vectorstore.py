"""
Comprehensive Rules vectorstore — RAG over the full MTG rules document.

Reference: Section 10.3 of PLAN.md.
"""

from __future__ import annotations

from pathlib import Path


def build_rules_vectorstore(
    comprehensive_rules_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """Build a FAISS vectorstore over the MTG Comprehensive Rules.

    Returns a FAISS vectorstore that can be used with
    `vectorstore.similarity_search(query, k=10)`.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    rules_text = Path(comprehensive_rules_path).read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". "],
    )
    chunks = splitter.split_text(rules_text)

    vectorstore = FAISS.from_texts(
        chunks,
        OpenAIEmbeddings(),
        metadatas=[
            {"source": "comprehensive_rules", "chunk_id": i}
            for i, _ in enumerate(chunks)
        ],
    )
    return vectorstore
