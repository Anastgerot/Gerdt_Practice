import os
from typing import List

SEPARATOR = "\n\n"

def read_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    

def write_paragraphs_by_lang(paragraphs: List[str], langs: List[str], output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unique_langs = set(langs)
    for lang in unique_langs:
        output_path = os.path.join(output_dir, f"{lang}.txt")
        if os.path.exists(output_path):
            os.remove(output_path)

    for lang, para in zip(langs, paragraphs):
        output_path = os.path.join(output_dir, f"{lang}.txt")
        with open(output_path, "a", encoding="utf-8") as out_file:
            out_file.write(para + SEPARATOR)