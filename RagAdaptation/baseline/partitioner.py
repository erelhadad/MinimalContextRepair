import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Tuple
try:
    from context_cite.context_partitioner import BaseContextPartitioner
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    class BaseContextPartitioner:
        def __init__(self, context: str) -> None:
            self.context = context


class TokenContextPartitioner(BaseContextPartitioner):
    """
    Partitions context into tokenizer-token spans using HF tokenizer offsets.

    Each "source" is the exact substring of the original context that corresponds
    to one tokenizer token (based on return_offsets_mapping).

    Ablation:
      - mask[i] == True  -> keep token substring
      - mask[i] == False -> either remove it, or replace with spaces (length preserving)

    Note: This creates num_sources ~ number of tokens in context, which can be huge.
    """

    def __init__(
        self,
        context: str,
        tokenizer,
        ablate_mode: str = "blank",  # "blank" or "remove"
    ) -> None:
        super().__init__(context)
        self.tokenizer = tokenizer
        if ablate_mode not in ("blank", "remove"):
            raise ValueError("ablate_mode must be 'blank' or 'remove'")
        self.ablate_mode = ablate_mode
        self._cache = {}
        self.split_context()

    def split_context(self) -> None:
        enc = self.tokenizer(
            self.context,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]

        # Keep only real spans
        spans: List[Tuple[int, int]] = [
            (s, e) for (s, e) in offsets
            if s is not None and e is not None and e > s
        ]

        self._cache["spans"] = spans

    @property
    def _spans(self) -> List[Tuple[int, int]]:
        spans = self._cache.get("spans")
        if spans is None:
            self.split_context()
            spans = self._cache["spans"]
        return spans

    @property
    def num_sources(self) -> int:
        return len(self._spans)

    def get_source(self, index: int) -> str:
        s, e = self._spans[index]
        return self.context[s:e]

    def get_context(self, mask: Optional[NDArray] = None) -> str:
        if mask is None:
            mask = np.ones(self.num_sources, dtype=bool)
        mask = np.asarray(mask, dtype=bool)

        if mask.shape[0] != self.num_sources:
            raise ValueError(f"Mask length {mask.shape[0]} != num_sources {self.num_sources}")

        out_parts: List[str] = []
        prev_end = 0

        for i, (s, e) in enumerate(self._spans):
            # Preserve any text between spans (usually none, but safe)
            if prev_end < s:
                out_parts.append(self.context[prev_end:s])

            tok_text = self.context[s:e]
            if mask[i]:
                out_parts.append(tok_text)
            else:
                if self.ablate_mode == "blank":
                    out_parts.append(" " * (e - s))
                else:  # remove
                    out_parts.append("")

            prev_end = e

        # Tail after last span
        out_parts.append(self.context[prev_end:])
        return "".join(out_parts)
