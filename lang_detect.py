import langid
import math
from typing import List, Tuple

def softmax(log_probs: List[float]) -> List[float]:
    max_log = max(log_probs)
    exps = [math.exp(lp - max_log) for lp in log_probs]
    total = sum(exps)
    return [e / total for e in exps]


def classify_paragraph(paragraph: str, min_confidence: float) -> Tuple[str, float]:
    ranked = langid.rank(paragraph)
    langs, log_probs = zip(*ranked[:5]) 
    probs = softmax(log_probs)
    lang, prob = langs[0], probs[0]
    if prob < min_confidence:
        return "unknown", prob
    return lang, prob