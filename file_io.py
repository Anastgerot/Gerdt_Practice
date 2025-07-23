import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

SEPARATOR = "\n"

def extract_lang_text(input_path: Path, output_path: Path) -> None:
    tree = ET.parse(input_path)
    root = tree.getroot()

    with output_path.open('w', encoding='utf-8') as out_file:
        for s in root.iter('s'):
            lang = s.attrib.get('lang')
            text = s.text.strip() if s.text else ''
            if lang and text:
                out_file.write(f"{lang} {text}\n")


def load_labeled_sentences(path: Path) -> List[Tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result = []
    for _, line in enumerate(lines):
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            lang, sentence = parts
            result.append((sentence, lang))
    return result
    

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


def prepare_text_file(input_path_str: str) -> Path:
    input_path = Path(input_path_str)
    if not input_path.exists():
        raise FileNotFoundError(f"File {input_path_str} not found.")

    if input_path.suffix.lower() == '.xml':
        output_txt = input_path.with_suffix('.txt')
        extract_lang_text(input_path, output_txt)
        print(f"[i] XML file converted to TXT: {output_txt}")
        return output_txt
    else:
        print(f"[i] File is already a TXT or compatible text format: {input_path}")
        return input_path
