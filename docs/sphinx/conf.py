from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

project = "opposition-agents-playing-mtg"
author = "Data Science Lab FHSWF"
copyright = "2026, Data Science Lab FHSWF"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
]

autosummary_generate = True
todo_include_todos = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

autodoc_mock_imports = [
    "torch",
    "neo4j",
    "sentence_transformers",
    "langchain",
    "langchain_community",
    "openai",
    "ollama",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]
