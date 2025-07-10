from pathlib import Path
import argparse
from file_io import read_file, write_sentences_by_lang
from text_processing import split_into_sentences, merge_same_language_sentences, attach_digit_dominant_sentences, refine_short_sentences
from fasttext_detect import classify_sentence
from models import ClassificationResult
import yaml
import logging.config

with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

logging.config.dictConfig(config["Logging"])
logger = logging.getLogger(__name__)

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect language(s) in multilingual text and split into files by language (sentence-level)."
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, default="Examples",
        help="Directory containing input text files (default: ./Examples)"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="Results",
        help="Directory to save output files (default: ./Results)"
    )
    parser.add_argument(
        "-c", "--min-confidence", type=float, default=0.9,
        help="Minimum confidence threshold for language classification (default: 0.9)"
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    text_files = list(input_dir.glob("*.txt"))

    if not text_files:
        logger.warning(f"No .txt files found in directory: {input_dir}")
        return

    for file_path in text_files:
        text = read_file(file_path)
        input_name = file_path.stem
        sentences = split_into_sentences(text)
        sentences = attach_digit_dominant_sentences(sentences)

        results: list[ClassificationResult] = [
            classify_sentence(sentence, min_confidence=args.min_confidence)
            for sentence in sentences]

        langs = [res.language for res in results]
        refine_short_sentences(sentences, langs)
        merged_sents, merged_langs = merge_same_language_sentences(sentences, langs)
        
        write_sentences_by_lang(merged_sents, merged_langs, output_dir, input_name)
        logger.info(f"{input_name}: Split into {len(set(merged_langs))} languages.")


if __name__ == "__main__":
    main()
