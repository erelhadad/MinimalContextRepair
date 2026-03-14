from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from langchain_core.documents import Document  # type: ignore
except ModuleNotFoundError:  # lightweight fallback for tests and plain txt flows
    @dataclass
    class Document:
        page_content: str
        metadata: dict


def load_pdf_documents(pdf_file: Path):
    """Load a single PDF as documents."""
    try:
        from langchain_community.document_loaders import PyPDFLoader  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        raise ModuleNotFoundError(
            "Loading PDF files requires langchain_community with PyPDFLoader installed."
        ) from exc

    loader = PyPDFLoader(str(pdf_file))
    return loader.load()


def load_txt_documents(txt_file: Path, encoding: str = "utf-8"):
    text = Path(txt_file).read_text(encoding=encoding, errors="ignore")
    return [Document(page_content=text, metadata={"source": str(txt_file)})]


def load_documents_any(path: str | Path):
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".pdf":
        return load_pdf_documents(p)
    if suf == ".txt":
        return load_txt_documents(p)
    raise ValueError(f"Unsupported file type: {suf} ({p})")


def combine_document_text(docs: Iterable[Document]) -> str:
    return "\n\n".join((d.page_content or "") for d in docs).strip()
