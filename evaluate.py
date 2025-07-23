import time
from pathlib import Path
from typing import List, Tuple, Callable

import pandas as pd
import pycountry
from sklearn.metrics import accuracy_score, f1_score
from text_processing import refine_short_sentences
from models import ClassificationResult
from lang_detectors import get_classifier
from file_io import load_labeled_sentences
from text_processing import attach_digit_dominant_sentences, refine_short_sentences
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def normalize_lang(code: str) -> str:
    code = code.lower()
    if code.startswith("language."):
        code = code.split(".", 1)[1]
    try:
        lang = pycountry.languages.lookup(code)
        return lang.alpha_2 if hasattr(lang, 'alpha_2') else code
    except LookupError:
        return code


def classify_sentences_from_file(
    file_path: Path,
    classifier: Callable[[str, float], ClassificationResult],
    min_confidence: float = 0.9,
    labeled: bool = False
) -> Tuple[List[str], List[str], List[str], float]:

    pairs = load_labeled_sentences(file_path)
    sentences = [p[0] for p in pairs]
    if labeled:
        ground_truth = [p[1] for p in pairs]
        sentences, ground_truth = attach_digit_dominant_sentences(sentences, ground_truth)
    else:
        sentences, _ = attach_digit_dominant_sentences(sentences)
        ground_truth = []

    start_time = time.perf_counter()
    results = [classifier(sentence, min_confidence=min_confidence) for sentence in sentences]
    end_time = time.perf_counter()

    predicted_langs = [res.language for res in results]
    refine_short_sentences(sentences, predicted_langs)

    return sentences, ground_truth, predicted_langs, end_time - start_time

def evaluate_classifier_on_dir(
    engine_name: str,
    input_dir: Path,
    min_confidence: float = 0.9
) -> List[Tuple[str, str, float, float, float]]:
    classifier: Callable[[str, float], ClassificationResult] = get_classifier(engine_name)
    results = []

    for file_path in input_dir.glob("*.txt"):
        input_name = file_path.stem
        sentences, ground_truth, predicted_langs, elapsed = classify_sentences_from_file(
            file_path, classifier, min_confidence, labeled=True)

        if not ground_truth:
            continue

        all_true = [normalize_lang(gt) for gt in ground_truth]
        all_pred = [normalize_lang(pred) for pred in predicted_langs]

        acc = accuracy_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred, average="macro")
        avg_time_ms = elapsed / len(sentences) * 1000

        results.append((engine_name, input_name, acc, f1, avg_time_ms))

    return results


def generate_markdown_report(results: List[Tuple[str, str, float, float, float]], output_dir: Path):
    df = pd.DataFrame(results, columns=["Engine", "File", "Accuracy", "Macro-F1", "Time per Sentence (ms)"])
    df = df.sort_values(by=["File", "Engine"])
    markdown = "# Language Classifier Comparison\n\n## Per-file Results\n\n"
    markdown += df.to_markdown(index=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.md"
    output_path.write_text(markdown, encoding="utf-8")
    logger.info(f"Report saved to: {output_path.absolute()}")


