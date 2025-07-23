import re
from typing import List, Tuple, Optional
import unicodedata

SHORT_SENTENCE_THRESHOLD = 30  # Максимальная длина короткого предложения
MIN_WORD_LENGTH = 4            # Слова короче игнорируются

def get_key_words(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r'\w+', text) if len(w) >= MIN_WORD_LENGTH]


def is_digit_dominant(text: str, threshold: float = 0.3) -> bool:
    letters = sum(1 for c in text if unicodedata.category(c).startswith("L"))
    total = sum(1 for c in text if not c.isspace())
    if total == 0:
        return False
    return (letters / total) < (1 - threshold) 


def attach_digit_dominant_sentences(
    sentences: List[str],
    labels: Optional[List[str]] = None
) -> Tuple[List[str], Optional[List[str]]]:
    new_sentences = []
    new_labels = [] if labels is not None else None
    i = 0

    while i < len(sentences):
        if is_digit_dominant(sentences[i]):
            if i + 1 < len(sentences):
                sentences[i + 1] = sentences[i] + " " + sentences[i + 1]
                if new_labels is not None:
                    pass
            elif new_sentences:
                new_sentences[-1] += " " + sentences[i]
                if new_labels is not None:
                    pass
        else:
            new_sentences.append(sentences[i])
            if new_labels is not None:
                new_labels.append(labels[i])
        i += 1

    return new_sentences, new_labels


def refine_short_sentences(sentences: List[str], langs: List[str]):

    for i, sent in enumerate(sentences):
        if len(sent) >= SHORT_SENTENCE_THRESHOLD:
            continue

        key_words = get_key_words(sent)
        if not key_words:
            continue

        for j, other in enumerate(sentences):
            if i == j or len(other) <= len(sent):
                continue

            if sent in other or any(
                re.search(r'\b' + re.escape(kw) + r'\b', other, flags=re.IGNORECASE)
                for kw in key_words
            ):
                langs[i] = langs[j]
                break

