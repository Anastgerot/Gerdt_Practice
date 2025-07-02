from pathlib import Path
import argparse
from file_io import read_file, write_paragraphs_by_lang, write_if_single_language_detected
from text_processing import split_paragraphs, refine_short_paragraphs
from lang_detect import classify_paragraphs
from text_processing import merge_same_language_paragraphs

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect language(s) in multilingual text and split into files by language."
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


def classify_and_extract(paragraphs, min_confidence):
    results = classify_paragraphs(paragraphs, min_confidence)
    langs = [r.language for r in results]
    confidences = [r.confidence for r in results]
    return langs, confidences


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    text_files = list(input_dir.glob("*.txt"))

    if not text_files:
        print(f"No .txt files found in directory: {input_dir}")
        return
    else:
        for file_path in text_files:
            text = read_file(file_path)
            paragraphs = split_paragraphs(text)

            langs, confidences = classify_and_extract(paragraphs, args.min_confidence)
            paragraphs, langs = merge_same_language_paragraphs(paragraphs, langs)
            langs, confidences = classify_and_extract(paragraphs, args.min_confidence)

            input_name = file_path.stem

            if write_if_single_language_detected(paragraphs, langs, confidences, args.output_dir, input_name):
                return

            refine_short_paragraphs(paragraphs, langs, args.min_confidence)
            write_paragraphs_by_lang(paragraphs, langs, args.output_dir, input_name)


if __name__ == "__main__":
    main()
