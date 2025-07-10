import fasttext
import re
from typing import List
from collections import Counter
from models import ClassificationResult

MODEL_PATH = "lid.176.ftz"
FASTTEXT_TOP_K = 1       # Сколько языков предсказывать
MIN_WORDS_FOR_RETRY = 3  # Минимум слов, чтобы запустить повторную классификацию
WINDOW_SIZE = 3          # Размер скользящего окна

model = fasttext.load_model(MODEL_PATH)

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def sliding_windows(words: List[str], window_size: int = WINDOW_SIZE) -> List[str]:
    return [" ".join(words[i:i + window_size]) for i in range(len(words) - window_size + 1)]


def classify_sentence(text: str, min_confidence: float) -> ClassificationResult:
    text = clean(text)

    if not text:
        return ClassificationResult(language="unknown", confidence=0.0, uncertain=True)

    # Первая попытка классификации
    labels, probs = model.predict(text, k=FASTTEXT_TOP_K)
    lang = labels[0].replace("__label__", "")
    conf = probs[0]
    uncertain = conf < min_confidence

    if not uncertain:
        return ClassificationResult(language=lang, confidence=conf, uncertain=False)

    # Повторная попытка при низкой уверенности
    words = text.split()
    if len(words) < MIN_WORDS_FOR_RETRY:
        return ClassificationResult(language=lang, confidence=conf, uncertain=True)

    windows = sliding_windows(words)
    lang_counts = Counter()

    for fragment in windows:
        frag_text = clean(fragment)
        labels, probs = model.predict(frag_text, k=FASTTEXT_TOP_K + 1)
        lang_ = labels[0].replace("__label__", "")
        conf_ = probs[0]
        if conf_ >= min_confidence:
            lang_counts[lang_] += 1

    if lang_counts:
        most_common_lang, count = lang_counts.most_common(1)[0]
        return ClassificationResult(
            language=most_common_lang,
            confidence=count / len(windows),
            uncertain=False
        )

    return ClassificationResult(language=lang, confidence=conf, uncertain=True)