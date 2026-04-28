"""
Comprehensive Rules vectorstore — RAG over the full MTG rules document.

The Comprehensive Rules text file lives in ``data/rules/`` (downloaded by
``scripts/fetch_rules.py`` from https://magic.wizards.com/en/rules).
``data/rules/latest.txt`` always points at the most recent revision.

Reference: Section 10.3 of PLAN.md.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "rules" / "latest.txt"
)


def build_rules_vectorstore(
    comprehensive_rules_path: str | Path | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """Build a FAISS vectorstore over the MTG Comprehensive Rules.

    Returns a FAISS vectorstore that can be used with
    `vectorstore.similarity_search(query, k=10)`.

    If ``comprehensive_rules_path`` is ``None``, defaults to
    ``data/rules/latest.txt``.  Run ``python scripts/fetch_rules.py`` to
    populate it.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    path = Path(comprehensive_rules_path or DEFAULT_RULES_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Comprehensive Rules file not found at {path}. "
            "Run: python scripts/fetch_rules.py"
        )
    rules_text = path.read_text(encoding="utf-8")

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
