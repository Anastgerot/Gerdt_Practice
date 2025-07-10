import re
from nltk.tokenize import sent_tokenize
from typing import List, Tuple
import unicodedata

SHORT_SENTENCE_THRESHOLD = 30  # Максимальная длина короткого предложения
MIN_WORD_LENGTH = 4            # Слова короче игнорируются

def split_into_sentences(text: str) -> List[str]:
    return [sent.strip() for sent in sent_tokenize(text) if sent.strip()]


def get_key_words(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r'\w+', text) if len(w) >= MIN_WORD_LENGTH]


def is_digit_dominant(text: str, threshold: float = 0.4) -> bool:
    letters = sum(1 for c in text if unicodedata.category(c).startswith("L"))
    total = sum(1 for c in text if not c.isspace())
    if total == 0:
        return False
    return (letters / total) < (1 - threshold) 


def attach_digit_dominant_sentences(sentences: List[str]) -> List[str]:
    new_sentences = []
    i = 0
    while i < len(sentences):
        if is_digit_dominant(sentences[i]):
            if i + 1 < len(sentences):
                sentences[i + 1] = sentences[i] + " " + sentences[i + 1]
            elif new_sentences:
                new_sentences[-1] += " " + sentences[i]
        else:
            new_sentences.append(sentences[i])
        i += 1
    return new_sentences


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


def merge_same_language_sentences(sentences: List[str], langs: List[str]) -> Tuple[List[str], List[str]]:
    if not sentences:
        return [], []

    merged_sentences = [sentences[0]]
    merged_langs = [langs[0]]

    for i in range(1, len(sentences)):
        if langs[i] == merged_langs[-1]:
            merged_sentences[-1] += " " + sentences[i]
        else:
            merged_sentences.append(sentences[i])
            merged_langs.append(langs[i])

    return merged_sentences, merged_langs
