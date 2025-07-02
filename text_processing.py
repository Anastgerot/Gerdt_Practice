import re
from typing import List, Tuple
from lang_detect import classify_paragraph
from lang_detect import UNKNOWN_LANG

SHORT_PARAGRAPH_THRESHOLD = 30  # Максимальная длина, при которой абзац считается коротким
REPEAT_EXTENSION_COUNT = 3      # Сколько раз повторять короткий текст для повышения уверенности

def split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r'\n\s*\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]


def get_key_words(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r'\w+', text) if len(w) > 3]


def refine_short_paragraphs(paragraphs: List[str], langs: List[str], min_confidence: float) -> None:
    for i, p in enumerate(paragraphs):
        if len(p) < SHORT_PARAGRAPH_THRESHOLD:
            key_words = get_key_words(p)
            found_match = False
            for j, big_p in enumerate(paragraphs):
                if i != j and len(big_p) > len(p):
                    if p in big_p:
                        langs[i] = langs[j]
                        found_match = True
                        break
                    elif any(re.search(r'\b' + re.escape(kw) + r'\b', big_p, flags=re.IGNORECASE) for kw in key_words):
                        langs[i] = langs[j]
                        found_match = True
                        break

            if found_match:
                continue 

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

    print(f"\nParagraphs before merging: {len(paragraphs)} → After: {len(merged_paragraphs)}")
    return merged_paragraphs, merged_langs