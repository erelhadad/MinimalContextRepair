import os
import shutil
from pathlib import Path
from typing import Iterable, List, Set

try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:  # pragma: no cover - optional in lightweight tests
    Chroma = None

try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
except ModuleNotFoundError:  # pragma: no cover - optional in lightweight tests
    PyPDFDirectoryLoader = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:  # pragma: no cover - optional in lightweight tests
    RecursiveCharacterTextSplitter = None

try:
    from langchain_core.documents import Document
except ModuleNotFoundError:  # pragma: no cover - optional in lightweight tests
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict

from RagAdaptation.core.documents import load_documents_any as core_load_documents_any
from RagAdaptation.core.paths import CHROMA_DIR, DATA_DIR

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = CHROMA_DIR
DATA_PATH = DATA_DIR


def load_documents(path: Path | str = DATA_PATH):
    if PyPDFDirectoryLoader is None:
        raise ModuleNotFoundError("load_documents requires langchain_community with PyPDFDirectoryLoader installed.")
    document_loader = PyPDFDirectoryLoader(str(path))
    return document_loader.load()


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 80,
):
    if RecursiveCharacterTextSplitter is None:
        raise ModuleNotFoundError("split_documents requires langchain_text_splitters installed.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,          # chars
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)


def _delete_by_sources(db: Chroma, sources: Set[str]) -> int:
    """
    Remove all rows whose metadata["source"] is in `sources`.
    This prevents 'old chunking' from persisting when you change chunk_size.
    """
    if not sources:
        return 0

    existing = db.get(include=["metadatas"])
    ids = existing.get("ids", [])
    metas = existing.get("metadatas", [])

    to_delete = []
    for _id, md in zip(ids, metas):
        if md and md.get("source") in sources:
            to_delete.append(_id)

    if to_delete:
        # langchain-chroma accepts delete(ids=...)
        db.delete(ids=to_delete)

    return len(to_delete)


def create_vector_store(
    chunks: List[Document],
    *,
    reset_db: bool = False,
    delete_existing_for_same_source: bool = True,
):
    """
    - If reset_db=True: wipe the entire DB and rebuild from scratch.
    - Else if delete_existing_for_same_source=True: delete all existing chunks
      that came from the same PDF source(s), then insert the new chunking.
    """
    if reset_db:
        clear_database()

    if Chroma is None:
        raise ModuleNotFoundError("create_vector_store requires langchain_chroma installed.")

    from RagAdaptation.get_embedding_function import get_embedding_function

    db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=get_embedding_function())

    chunks_with_ids = calculate_chunk_ids(chunks)

    if delete_existing_for_same_source:
        sources = {c.metadata.get("source") for c in chunks_with_ids if c.metadata.get("source")}
        deleted = _delete_by_sources(db, sources)
        print(f"Deleted {deleted} existing chunks for sources: {len(sources)}")

    # Now insert new chunks (fresh IDs for this run)
    new_ids = [c.metadata["id"] for c in chunks_with_ids]
    print(f"Adding {len(chunks_with_ids)} chunks to DB ...")
    db.add_documents(chunks_with_ids, ids=new_ids)


def load_documents_any(path: str):
    return core_load_documents_any(path)
