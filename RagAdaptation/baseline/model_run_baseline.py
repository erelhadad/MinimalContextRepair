import os
import sys
import shutil
import warnings
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from langchain_chroma import Chroma
from RagAdaptation.core.prompting import ChatPromptTemplate

#from RagAdaptation.document_handling.database_hadling import clear_database

# --- make `import RagAdaptation...` work even if you run this file directly ---
_THIS_FILE = Path(__file__).resolve()
_PKG_DIR = _THIS_FILE.parents[1]          # .../RagAdaptation
_PROJECT_ROOT = _PKG_DIR.parent           # .../RAG_EXP
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RagAdaptation.get_embedding_function import get_embedding_function
from RagAdaptation.core.documents import load_documents_any
from RagAdaptation.core.paths import CHROMA_DIR, DATA_DIR
from RagAdaptation.prompts_format import (
    TF_NO_CONTEXT_TEMPLATE,
    TF_RAG_TEMPLATE,
    normalize_true_false,
)

# Keep output clean
warnings.filterwarnings(
    "ignore",
    message=r"Using the tokenizer's special token policy `None` is deprecated.*",
    category=FutureWarning,
)

# ---- Paths (robust on Linux) ----
BASELINE_DIR = _THIS_FILE.parent
RAGADAPTATION_DIR = BASELINE_DIR.parent

CHROMA_PATH = CHROMA_DIR
PDF_FILENAME ="men-stronger.txt"

#"Is_sugar_addictive_text_only_no_header.pdf"
#"tall_man_example_article.txt"
#"vegan.txt"
PDF_PATH = DATA_DIR / PDF_FILENAME

mistral_models_path = Path.home().joinpath("mistral_models", "7B-Instruct-v0.3")
mistral_models_path.mkdir(parents=True, exist_ok=True)


def download_model(local_dir: Path):
    snapshot_download(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"],
        local_dir=local_dir,
    )

def load_mistral_tokenizer(local_dir: Path) -> MistralTokenizer:
    return MistralTokenizer.from_file(str(local_dir / "tokenizer.model.v3"))


def _get_raw_sp_tokenizer(mtok: MistralTokenizer):
    """Return underlying SentencePiece-like tokenizer (has .encode/.decode)."""
    if hasattr(mtok, "instruct_tokenizer") and hasattr(mtok.instruct_tokenizer, "tokenizer"):
        return mtok.instruct_tokenizer.tokenizer
    if hasattr(mtok, "tokenizer"):
        return mtok.tokenizer
    return None


def _sp_encode(raw_tok, text: str) -> list[int]:
    """
    Your SentencePieceTokenizer.encode() requires bos/eos args.
    We try several compatible call styles.
    """
    # Try keyword args
    try:
        out = raw_tok.encode(text, bos=False, eos=False)
        return out.tokens if hasattr(out, "tokens") else out
    except TypeError:
        pass

    # Try positional args
    try:
        out = raw_tok.encode(text, False, False)
        return out.tokens if hasattr(out, "tokens") else out
    except TypeError as e:
        raise TypeError(
            f"Could not encode with underlying tokenizer. "
            f"Tried encode(text, bos=False, eos=False) and encode(text, False, False). "
            f"Original error: {e}"
        )


def _sp_decode(raw_tok, ids: list[int]) -> str:
    out = raw_tok.decode(ids)
    return out if isinstance(out, str) else str(out)


def split_documents_by_mistral_tokens(docs, tokenizer: MistralTokenizer, chunk_tokens=50, overlap_tokens=20):
    """Token-based chunking using underlying SentencePiece tokenizer."""
    from langchain_core.documents import Document

    raw_tok = _get_raw_sp_tokenizer(tokenizer)
    if raw_tok is None:
        raise RuntimeError("Could not find underlying tokenizer with .encode/.decode")

    chunks = []
    for d in docs:
        text = d.page_content or ""
        ids = _sp_encode(raw_tok, text)
        n = len(ids)

        start = 0
        while start < n:
            end = min(start + chunk_tokens, n)
            chunk_ids = ids[start:end]
            chunk_text = _sp_decode(raw_tok, chunk_ids)

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=dict(getattr(d, "metadata", {}) or {}),
                )
            )

            if end == n:
                break
            start = max(0, end - overlap_tokens)

    return chunks


def mistral_generate_text(model: Transformer, tokenizer: MistralTokenizer, prompt_text: str, max_new_tokens=3) -> str:
    """
    Matches your installed signature:
    generate(encoded_prompts: List[List[int]], model: Transformer, *, max_tokens, temperature, ...)
    """
    req = ChatCompletionRequest(messages=[UserMessage(content=prompt_text)])
    enc = tokenizer.encode_chat_completion(req)
    prompt_ids = enc.tokens if hasattr(enc, "tokens") else enc

    eos_id: Optional[int] = None
    if hasattr(tokenizer, "instruct_tokenizer") and hasattr(tokenizer.instruct_tokenizer, "tokenizer"):
        eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id
    elif hasattr(tokenizer, "tokenizer"):
        eos_id = tokenizer.tokenizer.eos_id

    out_tokens, _ = generate(
        encoded_prompts=[prompt_ids],
        model=model,
        max_tokens=max_new_tokens,
        temperature=0.0,
        eos_id=eos_id,
    )

    seq = out_tokens[0]
    if len(seq) >= len(prompt_ids) and seq[:len(prompt_ids)] == prompt_ids:
        seq = seq[len(prompt_ids):]

    return tokenizer.decode(seq)



def rebuild_chroma_from_docs(docs, tokenizer: MistralTokenizer):
    """Clear and rebuild persistent Chroma DB under RagAdaptation/chroma_db."""
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)

    chunks = split_documents_by_mistral_tokens(docs, tokenizer, chunk_tokens=10, overlap_tokens=1)

    embedding_function = get_embedding_function()
    try:
        _ = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=str(CHROMA_PATH),
        )
    except TypeError:
        _ = Chroma.from_documents(
            documents=chunks,
            embedding_function=embedding_function,
            persist_directory=str(CHROMA_PATH),
        )

    return len(docs), len(chunks)


def show_run(title: str, prompt: str, raw_answer: str, context: str | None = None):
    ctx_chars = len(context.strip()) if context is not None else None
    raw = raw_answer.strip().replace("\n", " ")
    raw_short = (raw[:140] + " ...") if len(raw) > 140 else raw

    print(f"\n=== {title} ===")
    if ctx_chars is not None:
        print(f"Context chars: {ctx_chars}")
    print("Prompt:")
    print(repr(prompt.strip().replace("\n", " ")))
    print("Raw:", repr(raw_short))

    try:
        norm = normalize_true_false(raw_answer)
        print("Normalized:", norm)
    except Exception as e:
        print("Normalized: ERROR:", e)


def baseline(query_text: str):
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at: {PDF_PATH}\n"
            f"Expected it under: {DATA_DIR}\n"
            f"Fix: put {PDF_FILENAME} in RagAdaptation/data/ or change PDF_FILENAME."
        )

    download_model(mistral_models_path)
    tokenizer = load_mistral_tokenizer(mistral_models_path)
    model = Transformer.from_folder(mistral_models_path)

    # ---- NO CONTEXT ----
    prompt_no_ctx = ChatPromptTemplate.from_template(TF_NO_CONTEXT_TEMPLATE).format(query=query_text)
    ans_no_ctx_raw = mistral_generate_text(model, tokenizer, prompt_no_ctx, max_new_tokens=3)
    show_run("NO CONTEXT", prompt_no_ctx, ans_no_ctx_raw)

    # ---- FULL PDF CONTEXT (NO RETRIEVAL) ----
    docs = load_documents_any(PDF_PATH)
    print("pages:", len(docs))
    print("page lengths:", [len(d.page_content) for d in docs])
    full_context = "\n\n".join([d.page_content for d in docs]).strip()

    prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE)
    prompt_full = prompt_template.format(context=full_context, question=query_text)
    ans_full_raw = "false" if not full_context else mistral_generate_text(model, tokenizer, prompt_full, max_new_tokens=3)
    show_run("FULL PDF CONTEXT (NO RETRIEVAL)", prompt_full, ans_full_raw, context=full_context)

    # ---- RAG (TOP-1) ----
    n_docs, n_chunks = rebuild_chroma_from_docs(docs, tokenizer)
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=1)
    retrieved_context = results[0][0].page_content.strip() if results else ""

    print(f"\nDB built: docs={n_docs}, chunks={n_chunks}, retrieved_docs={len(results)}")

    prompt_rag = prompt_template.format(context=retrieved_context, query=query_text)
    ans_rag_raw = "false" if not retrieved_context else mistral_generate_text(model, tokenizer, prompt_rag, max_new_tokens=3)
    show_run("RETRIEVED CONTEXT (TOP-1)", prompt_rag, ans_rag_raw, context=retrieved_context)


if __name__ == "__main__":
    # q = "Is sugar considered an addictive substance?"
    q="Are shorter men can be considered more attractive? "
    #clear_database()
    # tokenizer = load_mistral_tokenizer(mistral_models_path)
    # req= ChatCompletionRequest(messages=[UserMessage(content=" ")])
    # print("req:")
    # print(req.messages[0])
    # print()
    # print()
    # end=tokenizer.encode_chat_completion(req)
    # print("end:")
    # print(end)

    baseline(q)


