from pathlib import Path
from typing import List
from collections import Counter

SEPARATOR = "\n\n"

def read_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File {file_path} not found.")
    return path.read_text(encoding="utf-8")

def write_paragraphs_by_lang(paragraphs: List[str], langs: List[str], output_dir: str, input_name: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    unique_langs = set(langs)
    for lang in unique_langs:
        file = output_path / f"{input_name}_{lang}.txt"
        if file.exists():
            file.unlink() 

    for lang, par in zip(langs, paragraphs):
        file = output_path / f"{input_name}_{lang}.txt"
        with file.open("a", encoding="utf-8") as out_file:
            out_file.write(par + SEPARATOR)


def write_if_single_language_detected(paragraphs, langs, confidences, output_dir, input_name: str) -> bool:
    lang_counts = Counter(langs)
    avg_conf = {
        lang: sum(c for l, c in zip(langs, confidences) if l == lang) / count
        for lang, count in lang_counts.items()
    }
    if len(lang_counts) == 1:
        only_lang = next(iter(lang_counts))
        if avg_conf[only_lang] >= 0.98:
            output_path = output_dir / f"{input_name}_{only_lang}.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("\n\n".join(paragraphs), encoding="utf-8")
            print(f"Only one language detected: {only_lang}.")
            return True
    return False