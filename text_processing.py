import re
from typing import List, Tuple
from lang_detect import classify_paragraph
from lang_detect import UNKNOWN_LANG
import logging

logger = logging.getLogger(__name__)

SHORT_PARAGRAPH_THRESHOLD = 30  # Максимальная длина, при которой абзац считается коротким
REPEAT_EXTENSION_COUNT = 3      # Сколько раз повторять короткий текст для повышения уверенности
MIN_WORD_LENGTH = 4  # Используются только слова длиной больше 3 символов

def split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r'\n\s*\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]


def get_key_words(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r'\w+', text) if len(w) >= MIN_WORD_LENGTH]


def refine_short_paragraphs(paragraphs: List[str], langs: List[str], min_confidence: float):
    for i, p in enumerate(paragraphs):
        if len(p) >= SHORT_PARAGRAPH_THRESHOLD:
            continue

        key_words = get_key_words(p)
        for j, big_p in enumerate(paragraphs):
            if i == j or len(big_p) <= len(p):
                continue

            if p in big_p or any(
                re.search(r'\b' + re.escape(kw) + r'\b', big_p, flags=re.IGNORECASE) 
                for kw in key_words
            ):
                langs[i] = langs[j]
                break
        else:
            extended_text = (p + " ") * REPEAT_EXTENSION_COUNT
            result = classify_paragraph(extended_text.strip(), min_confidence)
            if result.language != UNKNOWN_LANG:
                langs[i] = result.language


def merge_same_language_paragraphs(paragraphs: List[str], langs: List[str]) -> Tuple[List[str], List[str]]:
    if not paragraphs:
        return [], []
    merged_paragraphs = [paragraphs[0]]
    merged_langs = [langs[0]]

    for i in range(1, len(paragraphs)):
        if langs[i] == merged_langs[-1]:
            merged_paragraphs[-1] += " " + paragraphs[i]
        else:
            merged_paragraphs.append(paragraphs[i])
            merged_langs.append(langs[i])

    logger.info(f"Paragraphs before merging: {len(paragraphs)} → After: {len(merged_paragraphs)}")
    return merged_paragraphs, merged_langs