"""Load and validate documents/queries JSON datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import importlib
_logging_setup = importlib.import_module("src.logging-setup")
get_logger = _logging_setup.get_logger

logger = get_logger("data-loader")


@dataclass
class Document:
    id: str
    text: str


@dataclass
class Query:
    id: str
    query: str
    answer: str
    evidence: list[str]  # list of doc IDs


def load_documents(path: str | Path) -> list[Document]:
    """Load documents from JSON. Schema: [{"id": "doc_001", "text": "..."}]"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for item in raw:
        if "id" not in item or "text" not in item:
            logger.warning(f"Skipping invalid document entry: {item.get('id', 'unknown')}")
            continue
        docs.append(Document(id=item["id"], text=item["text"]))

    logger.info(f"Loaded {len(docs)} documents from {path}")
    return docs


def load_queries(path: str | Path) -> list[Query]:
    """Load queries from JSON.
    Schema: [{"id": "q_001", "query": "...", "answer": "...", "evidence": ["doc_001"]}]
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    queries = []
    for item in raw:
        if "id" not in item or "query" not in item:
            logger.warning(f"Skipping invalid query entry: {item.get('id', 'unknown')}")
            continue
        queries.append(Query(
            id=item["id"],
            query=item["query"],
            answer=item.get("answer", ""),
            evidence=item.get("evidence", []),
        ))

    logger.info(f"Loaded {len(queries)} queries from {path}")
    return queries
