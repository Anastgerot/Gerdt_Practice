from pathlib import Path
import argparse
from file_io import write_sentences_by_lang, prepare_text_file
from evaluate import classify_sentences_from_file, evaluate_classifier_on_dir, generate_markdown_report
from lang_detectors import get_classifier 
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
    parser.add_argument(
        "-e", "--engine", type=str, default="fasttext",
        choices=["fasttext", "langdetect", "langid", "lingua", "xlmroberta"],
        help="Language detection engine to use (default: fasttext)"
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    input_files = list(input_dir.glob("*.txt")) + list(input_dir.glob("*.xml"))

    if not input_files:
        logger.warning(f"No .txt or .xml files found in directory: {input_dir}")
        return

    for file_path in input_files:
        prepared_path = prepare_text_file(str(file_path))
        input_name = prepared_path.stem
        classifier = get_classifier(args.engine)

        sentences, _, langs, _ = classify_sentences_from_file(prepared_path, classifier, args.min_confidence, labeled=False)
        
        write_sentences_by_lang(sentences, langs, output_dir, input_name)
        logger.info(f"{input_name}: split into {len(set(langs))} languages")

    engines = ["fasttext", "langdetect", "langid", "lingua", "xlmroberta"]
    all_results = []

    for engine in engines:
        logger.info(f"Evaluating engine: {engine}")
        engine_results = evaluate_classifier_on_dir(
            engine_name=engine,
            input_dir=input_dir,
            min_confidence=args.min_confidence
        )
        all_results.extend(engine_results)

    if all_results:
        generate_markdown_report(all_results, output_dir)
    else:
        logger.info("No labeled .txt files found for evaluation.")


if __name__ == "__main__":
    main()
