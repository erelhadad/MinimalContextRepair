from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Set, Tuple,List,Dict,Optional,Any,Sequence
import numpy as np

from RagAdaptation.prompts_format import TF_RAG_TEMPLATE, TF_RAG_TEMPLATE_A2T

def get_mask_prompt_template(change_template_contextCite: bool):
    if change_template_contextCite:
        return ChatPromptTemplate.from_template(TF_RAG_TEMPLATE_A2T)
    return ChatPromptTemplate.from_template(TF_RAG_TEMPLATE)

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


@dataclass(frozen=True)
class WordUnit:
    """ Represent one replaceable word in the original text.
    piece_index: tells us which peice in the piece list should be edited. Cosiders spaces as a piece
    """
    word: int
    text: str
    start: int
    end: int
    piece_index: int # tells us which peice in the piece list should be edited


class InterventionMode(Enum):
    MASK_TOKEN = 1
    MASK_WORD = 2
    REPLACEMENT_NEUTRAL_WORD = 3
    REPLACEMENT_ANTONYM_WORD = 4

"""
This creates three categories:

word-like pieces: "sugar", "classified", "two-state"
whitespace pieces: " ", "\n"
punctuation pieces: ".", ",", "("

"""
_WORD_PIECE_RE = re.compile(r"\w+(?:[-']\w+)*|\s+|[^\w\s]", re.UNICODE)
_WORD_ONLY_RE = re.compile(r"^\w+(?:[-']\w+)*$", re.UNICODE)


def coerce_intervention_mode(mode: InterventionMode | str) -> InterventionMode:
    if isinstance(mode, InterventionMode):
        return mode

    if isinstance(mode, str):
        key = mode.strip().upper()
        aliases = {"MASK": InterventionMode.MASK_TOKEN,
            "MASK_TOKEN": InterventionMode.MASK_TOKEN,
            "TOKEN_MASK": InterventionMode.MASK_TOKEN,

            "MASK_WORD": InterventionMode.MASK_WORD,
            "WORD_MASK": InterventionMode.MASK_WORD,

            "NEUTRAL": InterventionMode.REPLACEMENT_NEUTRAL_WORD,
            "REPLACEMENT_NEUTRAL_WORD": InterventionMode.REPLACEMENT_NEUTRAL_WORD,
            "NEUTRAL_WORD": InterventionMode.REPLACEMENT_NEUTRAL_WORD,

            "ANTONYM": InterventionMode.REPLACEMENT_ANTONYM_WORD,
            "REPLACEMENT_ANTONYM_WORD": InterventionMode.REPLACEMENT_ANTONYM_WORD,
            "ANTONYM_WORD": InterventionMode.REPLACEMENT_ANTONYM_WORD,
        }
        if key in aliases:
            return aliases[key]

    raise ValueError(f"Unsupported intervention mode: {mode!r}")


def split_context_to_word_units(context: str) -> tuple[list[str], list[WordUnit]]:
    """
    Split context into stable string pieces and word units.

    Important:
    - pieces preserve the original context exactly: ''.join(pieces) == context
    - WordUnit.word is a dense word id: 0, 1, 2, ...
    - punctuation and whitespace are preserved as pieces but are not WordUnits
    """
    pieces: list[str] = []
    word_units: list[WordUnit] = []

    pos = 0
    word_id = 0
    # regex for all pieces words
    for m in _WORD_PIECE_RE.finditer(context):
        if m.start() > pos:
            pieces.append(context[pos:m.start()])

        piece_index = len(pieces)
        text = m.group(0)
        pieces.append(text)

        # regex only for identifying words
        if _WORD_ONLY_RE.match(text):
            word_units.append(WordUnit(word=word_id,text=text,start=int(m.start()),end=int(m.end()),piece_index=piece_index,))
            word_id += 1

        pos = m.end()

    if pos < len(context):
        pieces.append(context[pos:])

    if "".join(pieces) != context:
        raise RuntimeError("Internal error: word-piece split did not preserve context exactly.")

    return pieces, word_units


def _find_single_word_unit_for_span(word_units: Sequence[WordUnit], span: Tuple[int, int],) -> WordUnit:
    """
    Map one source span to exactly one WordUnit.

    This intentionally does NOT aggregate. If a span overlaps zero words or more
    than one word, we fail because the caller did not provide clean word-level
    sources.
    """
    s, e = int(span[0]), int(span[1])

    overlaps: list[WordUnit] = []
    for unit in word_units:
        if min(e, unit.end) > max(s, unit.start):
            overlaps.append(unit)

    if len(overlaps) != 1:
        raise ValueError("Expected each source offset to overlap exactly one word. " f"Got {len(overlaps)} overlaps for span={span}.This means the scores are not clean word-level scores.")

    return overlaps[0]


def build_word_candidates_no_aggregation(*,context: str, scores_vec: Optional[np.ndarray],source_offsets: Optional[Sequence[Tuple[int, int]]],
    rng: Optional[np.random.Generator],) -> tuple[list[str], list[WordUnit], list[int], Optional[list[float]]]:
    """
    Build word-level candidate order without aggregating scores.

    Valid cases:
    1. scores_vec is None:
       random word order over all WordUnits.

    2. len(scores_vec) == len(word_units):
       scores are already aligned 1:1 with our WordUnit list.

    3. source_offsets is provided and len(scores_vec) == len(source_offsets):
       each source offset must map to exactly one WordUnit, and no two scores may
       map to the same WordUnit.

    Invalid:
    - token-level scores where multiple source offsets map to the same word.
      This function raises instead of aggregating.
    """
    pieces, word_units = split_context_to_word_units(context)

    if rng is None:
        rng = np.random.default_rng()

    if scores_vec is None:
        order = [int(i) for i in rng.permutation(len(word_units))]
        return pieces, word_units, order, None

    scores_vec = np.asarray(scores_vec, dtype=np.float32)

    # Case 1: scores already directly correspond to our word_units.
    if len(scores_vec) == len(word_units):
        order = [int(i) for i in np.argsort(scores_vec)[::-1]]
        pick_scores = [float(scores_vec[i]) for i in order]
        return pieces, word_units, order, pick_scores

    # Case 2: scores correspond to externally provided word offsets.
    if source_offsets is None:
        raise ValueError(
            f"Word-level intervention received {len(scores_vec)} scores, "
            f"but context has {len(word_units)} words and source_offsets=None. "
            "To avoid aggregation, pass word-level source_offsets or word-aligned scores."
        )

    if len(scores_vec) != len(source_offsets):
        raise ValueError(f"len(scores_vec)={len(scores_vec)} but len(source_offsets)={len(source_offsets)}"
        )

    word_id_to_score: dict[int, float] = {}

    for src_i, span in enumerate(source_offsets):
        unit = _find_single_word_unit_for_span(word_units, span)

        if unit.word in word_id_to_score:
            raise ValueError(
                "Multiple scored sources map to the same word. "
                f"word_id={unit.word}, word_text={unit.text!r}. "
                "Refusing to aggregate scores."
            )

        word_id_to_score[int(unit.word)] = float(scores_vec[src_i])

    order = sorted(word_id_to_score.keys(), key=lambda wid: word_id_to_score[wid], reverse=True)
    pick_scores = [float(word_id_to_score[wid]) for wid in order]

    return pieces, word_units, order, pick_scores


def _lookup_replacement(unit: WordUnit,replacement_map: Optional[Mapping[Any, str]],) -> Optional[str]:
    """
    Supports maps keyed by:
    - word id: int
    - exact surface form: str
    - lowercase surface form: str
    """
    if replacement_map is None:
        return None

    keys = [
        unit.word,
        unit.text,
        unit.text.lower(),
        unit.text.strip(),
        unit.text.strip().lower(),
    ]

    for key in keys:
        if key in replacement_map:
            repl = replacement_map[key]
            if repl is not None and str(repl).strip():
                return str(repl).strip()

    return None


def _preserve_case(original: str, replacement: str) -> str:
    if not replacement:
        return replacement

    if original.isupper():
        return replacement.upper()

    if len(original) > 0 and original[0].isupper():
        return replacement[:1].upper() + replacement[1:]

    return replacement


def _word_has_valid_intervention(unit: WordUnit,*,mode: InterventionMode,replacement_map: Optional[Mapping[Any, str]],
) -> bool:

    if mode in {InterventionMode.MASK_TOKEN, InterventionMode.MASK_WORD}:
        return True

    repl = _lookup_replacement(unit, replacement_map)
    if repl is None:
        return False

    return repl.strip().lower() != unit.text.strip().lower()


def _filter_word_order_by_available_intervention(*,word_units: Sequence[WordUnit],order: Sequence[int],pick_scores: Optional[Sequence[float]],
    mode: InterventionMode,replacement_map: Optional[Mapping[Any, str]],) -> tuple[list[int], Optional[list[float]]]:
    """
    For replacement modes, skip words with no available replacement.
    This avoids wasting model calls on no-op interventions.
    """
    unit_by_id = {int(u.word): u for u in word_units}

    new_order: list[int] = []
    new_scores: list[float] = []

    for pos, wid in enumerate(order):
        unit = unit_by_id[int(wid)]

        if not _word_has_valid_intervention(unit,mode=mode,replacement_map=replacement_map,):
            continue

        new_order.append(int(wid))

        if pick_scores is not None:
            new_scores.append(float(pick_scores[pos]))

    return new_order, None if pick_scores is None else new_scores


def build_context_with_word_interventions(*,pieces: Sequence[str],word_units: Sequence[WordUnit],selected_word_ids: Set[int],
    mode: InterventionMode,replacement_map: Optional[Mapping[Any, str]] = None,) -> str:
    """
    Build a new context from stable pieces.

    This never uses post-replacement character offsets.
    """
    mode = coerce_intervention_mode(mode)

    out = list(pieces)
    unit_by_id = {int(u.word): u for u in word_units}

    for wid in selected_word_ids:
        unit = unit_by_id[int(wid)]

        if mode == InterventionMode.MASK_WORD:
            # Same-length word-level blanking. Keeps context length stable.
            out[unit.piece_index] = " " * len(unit.text)

        elif mode in {
            InterventionMode.REPLACEMENT_NEUTRAL_WORD,
            InterventionMode.REPLACEMENT_ANTONYM_WORD,
        }:
            repl = _lookup_replacement(unit, replacement_map)
            if repl is None:
                # Should normally be filtered before scoring.
                continue

            out[unit.piece_index] = _preserve_case(unit.text, repl)

        elif mode == InterventionMode.MASK_TOKEN:
            raise ValueError("build_context_with_word_interventions does not handle MASK_TOKEN.")

        else:
            raise ValueError(f"Unsupported intervention mode: {mode}")

    return "".join(out)


def build_context_with_word_interventions_metadata(*,pieces: Sequence[str],word_units: Sequence[WordUnit],selected_word_ids: Set[int],
    mode: InterventionMode,replacement_map: Optional[Mapping[Any, str]] = None,) -> tuple[str, Dict[int, Dict[str, Any]]]:
    """
    Build a word-intervened context and return per-word replacement metadata.
    """
    mode = coerce_intervention_mode(mode)
    context = build_context_with_word_interventions(
        pieces=pieces,
        word_units=word_units,
        selected_word_ids=selected_word_ids,
        mode=mode,replacement_map=replacement_map,
    )

    metadata: Dict[int, Dict[str, Any]] = {}
    unit_by_id = {int(u.word): u for u in word_units}
    for wid in selected_word_ids:
        unit = unit_by_id[int(wid)]
        entry: Dict[str, Any] = {
            "original": unit.text,
            "original_span": [int(unit.start), int(unit.end)],
        }
        if mode == InterventionMode.MASK_WORD:
            entry["replacement"] = " " * len(unit.text)
        elif mode in {InterventionMode.REPLACEMENT_NEUTRAL_WORD, InterventionMode.REPLACEMENT_ANTONYM_WORD}:
            repl = _lookup_replacement(unit, replacement_map)
            entry["replacement"] = None if repl is None else _preserve_case(unit.text, repl)
        metadata[int(wid)] = entry

    return context, metadata


def iter_word_intervention_prompts_iterative_chunks(document: str,query: str,*,pieces: Sequence[str],word_units: Sequence[WordUnit],ordered_word_ids: Sequence[int],
    intervention_mode: InterventionMode,replacement_map: Optional[Mapping[Any, str]] = None,change_template_contextCite: bool = False,chunk_size: int = 32,):
    """
    Yield prompts for cumulative word-level interventions.

    Step i applies interventions to:
        ordered_word_ids[0], ..., ordered_word_ids[i]

    No character offsets are used after replacement.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    mode = coerce_intervention_mode(intervention_mode)
    if mode == InterventionMode.MASK_TOKEN:
        raise ValueError("Use iter_masked_prompts_iterative_chunks for MASK_TOKEN.")

    prompt_template = get_mask_prompt_template(change_template_contextCite)

    selected_word_ids: set[int] = set()
    prompt_chunk: list[str] = []
    context_chunk: list[str] = []

    for wid in ordered_word_ids:
        selected_word_ids.add(int(wid))

        intervened_context = build_context_with_word_interventions(pieces=pieces,word_units=word_units,selected_word_ids=selected_word_ids, mode=mode,replacement_map=replacement_map,
        )

        context_chunk.append(intervened_context)

        if change_template_contextCite:
            prompt_chunk.append(prompt_template.format(context=intervened_context, query=query))
        else:
            prompt_chunk.append(prompt_template.format(context=intervened_context, question=query))

        if len(prompt_chunk) >= chunk_size:
            yield prompt_chunk, context_chunk
            prompt_chunk = []
            context_chunk = []

    if prompt_chunk:
        yield prompt_chunk, context_chunk


def create_word_intervention_prompts_iterative(document: str,query: str,*,pieces: Sequence[str],word_units: Sequence[WordUnit],ordered_word_ids: Sequence[int],
    intervention_mode: InterventionMode,replacement_map: Optional[Mapping[Any, str]] = None,change_template_contextCite: bool = False,):

    batch: list[str] = []
    context_list: list[str] = []

    for prompt_chunk, context_chunk in iter_word_intervention_prompts_iterative_chunks(document,query,pieces=pieces,word_units=word_units,ordered_word_ids=ordered_word_ids,intervention_mode=intervention_mode,replacement_map=replacement_map,
        change_template_contextCite=change_template_contextCite,chunk_size=max(1, len(ordered_word_ids) or 1),):

        batch.extend(prompt_chunk)
        context_list.extend(context_chunk)

    return batch, context_list
