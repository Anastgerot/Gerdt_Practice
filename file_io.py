from pathlib import Path
from typing import List
from collections import Counter
import logging

logger = logging.getLogger(__name__)

SEPARATOR = "\n"

def read_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File {file_path} not found.")
    return path.read_text(encoding="utf-8")


def clear_output_files_for_input(output_dir: Path, input_name: str) -> None:
    for old_file in output_dir.glob(f"{input_name}_*.txt"):
        try:
            old_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete {old_file}: {e}")


def write_sentences_by_lang(sentences: List[str], langs: List[str], output_dir: str, input_name: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clear_output_files_for_input(output_path, input_name)

    for lang, sent in zip(langs, sentences):
        file = output_path / f"{input_name}_{lang}.txt"
        with file.open("a", encoding="utf-8") as out_file:
            out_file.write(sent + SEPARATOR)
