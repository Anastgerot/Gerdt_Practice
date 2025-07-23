import fasttext
from langdetect import detect_langs
import langid
from lingua import LanguageDetectorBuilder
import re
from typing import List, Callable
from collections import Counter, defaultdict
from models import ClassificationResult
from xlm_roberta_detector import detect_xlmroberta 

MODEL_PATH = "lid.176.ftz"
FASTTEXT_TOP_K = 2       # Сколько языков предсказывать
MIN_WORDS_FOR_RETRY = 3  # Минимум слов, чтобы запустить повторную классификацию
WINDOW_SIZE = 3          # Размер скользящего окна
CHAR_WINDOW_SIZE = 4     # Размер скользящего окна для иероглифов (по символам)
CHAR_STEP = 1

_fasttext_model = fasttext.load_model(MODEL_PATH)
_lingua_detector = LanguageDetectorBuilder.from_all_languages().build()

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def sliding_char_windows(text: str, window_size: int = CHAR_WINDOW_SIZE, step: int = CHAR_STEP):
    return [text[i:i+window_size] for i in range(0, len(text) - window_size + 1, step)]


def sliding_windows(words: List[str], window_size: int = WINDOW_SIZE) -> List[str]:
    return [" ".join(words[i:i + window_size]) for i in range(len(words) - window_size + 1)]


def contains_cjk(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))


def retry_with_chars_fasttext(
    text: str,
    original_lang: str,
    original_conf: float,
    min_confidence: float
) -> ClassificationResult:
    if len(text.strip()) < CHAR_WINDOW_SIZE:
        return ClassificationResult(language=original_lang, confidence=original_conf, uncertain=True)

    windows = sliding_char_windows(text)
    lang_counts = Counter()
    lang_conf_sum = defaultdict(float)

    for fragment in windows:
        frag_text = clean(fragment)
        labels, probs = _fasttext_model.predict(frag_text, k=FASTTEXT_TOP_K)

        for j in range(len(labels)):
            lang = labels[j].replace("__label__", "")
            conf = probs[j]
            lang_counts[lang] += 1
            lang_conf_sum[lang] += conf

    if lang_counts:
        lang_avg_conf = {
            lang: lang_conf_sum[lang] / count
            for lang, count in lang_counts.items()
        }

        chosen_lang = max(lang_avg_conf, key=lang_avg_conf.get)
        final_conf = lang_avg_conf[chosen_lang]

        return ClassificationResult(
            language=chosen_lang,
            confidence=final_conf,
            uncertain=final_conf < min_confidence
        )

    return ClassificationResult(language=original_lang, confidence=original_conf, uncertain=True)


def retry_with_windows_fasttext(text: str, original_lang: str, original_conf: float, min_confidence: float) -> ClassificationResult:
    words = text.split()
    if len(words) < MIN_WORDS_FOR_RETRY:
        return ClassificationResult(language=original_lang, confidence=original_conf, uncertain=True)

    windows = sliding_windows(words)
    lang_counts = Counter()
    lang_conf_sum = defaultdict(float)

    for i, fragment in enumerate(windows):
        frag_text = clean(fragment)
        labels, probs = _fasttext_model.predict(frag_text, k=FASTTEXT_TOP_K)

        for j in range(len(labels)):
            lang = labels[j].replace("__label__", "")
            conf = probs[j]
            lang_counts[lang] += 1
            lang_conf_sum[lang] += conf

    if lang_counts:
        max_count = max(lang_counts.values())
        top_langs = [lang for lang, count in lang_counts.items() if count == max_count]

        lang_avg_conf = {}
        for lang in top_langs:
            avg_conf = lang_conf_sum[lang] / lang_counts[lang]
            lang_avg_conf[lang] = avg_conf

        chosen_lang = max(lang_avg_conf, key=lang_avg_conf.get)
        final_conf = lang_avg_conf[chosen_lang]

        return ClassificationResult(
            language=chosen_lang,
            confidence=final_conf,
            uncertain=final_conf < min_confidence
        )

    return ClassificationResult(language=original_lang, confidence=original_conf, uncertain=True)


def detect_fasttext(text: str, min_confidence: float) -> ClassificationResult:
    text = clean(text)
    if not text:
        return ClassificationResult(language="unknown", confidence=0.0, uncertain=True)

    labels, probs = _fasttext_model.predict(text, k=FASTTEXT_TOP_K)
    lang = labels[0].replace("__label__", "")
    conf = probs[0]
    if conf >= min_confidence:
        return ClassificationResult(language=lang, confidence=conf, uncertain=False)

    if contains_cjk(text):
        return retry_with_chars_fasttext(text, lang, conf, min_confidence)
    else:
        return retry_with_windows_fasttext(text, lang, conf, min_confidence)


def detect_langdetect(text: str, min_confidence: float) -> ClassificationResult:
    try:
        langs = detect_langs(text)
        if not langs:
            return ClassificationResult(language="unknown", confidence=0.0, uncertain=True)

        lang = langs[0].lang
        conf = langs[0].prob
        if conf >= min_confidence:
            return ClassificationResult(language=lang, confidence=conf, uncertain=False)

        return retry_with_windows_fasttext(text, lang, conf, min_confidence)
    except Exception:
        return ClassificationResult(language="unknown", confidence=0.0, uncertain=True)


def detect_langid(text: str, min_confidence: float) -> ClassificationResult:
    lang, conf = langid.classify(text)
    if conf >= min_confidence:
        return ClassificationResult(language=lang, confidence=conf, uncertain=False)

    return retry_with_windows_fasttext(text, lang, conf, min_confidence)


def detect_lingua(text: str, min_confidence: float) -> ClassificationResult:
    try:
        values = _lingua_detector.compute_language_confidence_values(text)
        if not values:
            return ClassificationResult(language="unknown", confidence=0.0, uncertain=True)

        top = values[0]
        lang = str(top.language).lower()
        conf = top.value

        return ClassificationResult(language=lang, confidence=conf, uncertain=conf < min_confidence)
    except Exception:
        return ClassificationResult(language="unknown", confidence=0.0, uncertain=True)


def get_classifier(engine_name: str) -> Callable[[str, float], ClassificationResult]:
    if engine_name == "fasttext":
        return detect_fasttext
    elif engine_name == "langdetect":
        return detect_langdetect
    elif engine_name == "langid":
        return detect_langid
    elif engine_name == "lingua":
        return detect_lingua
    elif engine_name == "xlmroberta":
        return detect_xlmroberta
    else:
        raise ValueError(f"Unknown engine: {engine_name}")