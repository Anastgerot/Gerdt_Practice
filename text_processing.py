import re
from typing import List
from lang_detect import classify_paragraph

def split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r'\n\s*\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]


def get_key_words(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r'\w+', text) if len(w) > 3]


def refine_short_paragraphs(paragraphs: List[str], langs: List[str], min_confidence: float) -> None:
    for i, p in enumerate(paragraphs):
        if len(p) < 30:
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

            extended_text = (p + " ") * 3
            lang, conf = classify_paragraph(extended_text.strip(), min_confidence)
            if lang != "unknown":
                langs[i] = lang