import re


def normalize_true_false(text: str) -> str:
    s = text.strip()

    # מפרידה camelCase: trueHuman -> true Human
    s = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s)

    # מאחדת גרש רגיל וגרש טיפוגרפי
    s = s.replace("’", "'")

    # מסירה prefixes נפוצים בתחילת הטקסט
    s = re.sub(r'^\s*(answer|response|output)\s*:\s*', '', s, flags=re.IGNORECASE)

    # קודם כל בודקות שלילה מפורשת: is not / isn't
    neg = re.search(r"\b(?:is\s+not|isn't)\s+(true|false|yes|no)\b", s, flags=re.IGNORECASE)
    if neg:
        val = neg.group(1).lower()
        if val in {"true", "yes"}:
            return "false"
        return "true"

    # אחר כך מחפשות true/false/yes/no בכל מקום במשפט
    m = re.search(r"\b(true|false|yes|no)\b", s, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Model did not return true/false/yes/no. Got: {text!r}")

    val = m.group(1).lower()
    if val in {"true", "yes"}:
        return "true"
    return "false"


# Minimal-pair prompts (same structure, only context differs)
TF_NO_CONTEXT_TEMPLATE = """Answer with exactly one word: true or false.


question: {question}
Answer:
"""

TF_RAG_TEMPLATE = """Answer the question with exactly one word: true or false. 
Use ONLY the context. 
Context:
{context}.

question: {question}
Answer:
"""

TF_RAG_TEMPLATE_EMPTY = """Answer the question with exactly one word: true or false. 


question: {question}
Answer:
"""

TF_RAG_TEMPLATE_A2T = """Answer the question with exactly one word: true or false.
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
