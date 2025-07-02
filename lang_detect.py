import langid
import numpy as np
from typing import List, Tuple
from lingua import LanguageDetectorBuilder
from models import ClassificationResult

UNKNOWN_LANG = "unknown"
TOP_LANG_CANDIDATES = 5 

lingua_detector = LanguageDetectorBuilder.from_all_languages().build()

def softmax(log_probs: List[float]) -> List[float]:
    log_probs = np.array(log_probs)
    exps = np.exp(log_probs - np.max(log_probs))
    return (exps / exps.sum()).tolist()


def classify_paragraph(paragraph: str, min_confidence: float) -> Tuple[str, float]:
    info = lingua_detector.compute_language_confidence_values(paragraph)
    if info:
        lang, prob = info[0].language.iso_code_639_1.name.lower(), info[0].value
        if prob >= min_confidence:
            return ClassificationResult(language=lang, confidence=prob)

    # Дополнительный вариант: используем langid, берём топ-5 наиболее вероятных языков
    ranked = langid.rank(paragraph)
    langs, log_probs = zip(*ranked[:TOP_LANG_CANDIDATES])
    probs = softmax(log_probs)
    if probs[0] < min_confidence:
        return ClassificationResult(language=UNKNOWN_LANG, confidence=probs[0])
    return ClassificationResult(language=langs[0], confidence=probs[0])


def classify_paragraphs(paragraphs: List[str], min_confidence: float) -> List[ClassificationResult]:
    return [classify_paragraph(p, min_confidence) for p in paragraphs]
