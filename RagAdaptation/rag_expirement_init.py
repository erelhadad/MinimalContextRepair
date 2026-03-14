import argparse

try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Chroma = None

try:
    from RagAdaptation.core.prompting import ChatPromptTemplate
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ChatPromptTemplate = None

try:
    from langchain_huggingface import HuggingFacePipeline
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    HuggingFacePipeline = None

from RagAdaptation import prompts_format
from RagAdaptation.core.paths import CHROMA_DIR, DATA_DIR
from RagAdaptation.document_handling.database_hadling import (
    clear_database,
    create_vector_store,
    load_documents,
    split_documents,
)

CHROMA_PATH = str(CHROMA_DIR)
DATA_PATH = str(DATA_DIR)
K = 5


def _require_runtime_deps():
    if Chroma is None or ChatPromptTemplate is None or HuggingFacePipeline is None:
        raise ModuleNotFoundError(
            "rag_expirement_init requires langchain_chroma, langchain_core, and langchain_huggingface."
        )


def query_rag(query_text: str):
    _require_runtime_deps()
    from RagAdaptation.get_embedding_function import get_embedding_function
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=K)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(prompts_format.TF_RAG_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = HuggingFacePipeline.from_model_id(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 5,
            "do_sample": False,
            "return_full_text": False,
            "temperature": 0,
        },
    )
    response_text = model.invoke(prompt)
    response_text = prompts_format.normalize_true_false(response_text)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


def query_llm_without_context(query: str):
    _require_runtime_deps()
    model = HuggingFacePipeline.from_model_id(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 5,
            "do_sample": False,
            "return_full_text": False,
            "temperature": 0,
        },
    )
    prompt = ChatPromptTemplate.from_template(prompts_format.TF_NO_CONTEXT_TEMPLATE).format(question=query)
    raw_ans = model.invoke(prompt)
    return prompts_format.normalize_true_false(raw_ans)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Clear the database before adding documents")
    parser.add_argument("--document_path", type=str, default=DATA_PATH, help="Path to the directory containing PDF documents")
    parser.add_argument("--query", type=str, default="", help="Query text")
    parser.add_argument("--use_context", action="store_true", default=False, help="")
    args = parser.parse_args()
    if args.reset:
        clear_database()

    print("-----------------")
    q = "Is sugar an addictive substance?"
    print("-----------------")
    if args.query:
        q = args.query
    print("Query:", q)
    print("model response: ")
    if args.use_context:
        documents = load_documents(args.document_path)
        chunks = split_documents(documents)
        create_vector_store(chunks)
        response = query_rag(q)
    else:
        response = query_llm_without_context(q)
        print(response)


if __name__ == "__main__":
    main()
