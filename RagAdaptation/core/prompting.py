from __future__ import annotations

try:
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    class ChatPromptTemplate:
        def __init__(self, template: str):
            self.template = template

        @classmethod
        def from_template(cls, template: str):
            return cls(template)

        def format(self, **kwargs):
            return self.template.format(**kwargs)
