import re

def normalize_true_false(text: str) -> str:
    m = re.search(r"\b(true|false)\b", text.strip().lower())
    if not m:
        raise ValueError(f"Model did not return true/false. Got: {text!r}")
    return m.group(1)


# Minimal-pair prompts (same structure, only context differs)
TF_NO_CONTEXT_TEMPLATE = """Answer with exactly one word: true or false.


question: {question}
Answer:
"""

TF_RAG_TEMPLATE = """Answer with exactly one word: true or false.
Use ONLY the context. 

Context:
{context}

question: {question}
Answer: 
"""
TF_RAG_TEMPLATE_A2T = """Answer with exactly one word: true or false.
Use ONLY the context. 

Context:
{context}

question: {query}
Answer:
"""


# (kept for compatibility if you use these elsewhere)
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""
