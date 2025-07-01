import argparse
from file_io import read_file, write_paragraphs_by_lang
from text_processing import split_paragraphs, refine_short_paragraphs
from lang_detect import classify_paragraph

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Определение языка для многоязычных текстов и разбивание по файлам."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Имя входного файла с текстом"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=".",
        help="Каталог для сохранения результатов (по умолчанию текущий каталог)"
    )
    parser.add_argument(
        "-c", "--min-confidence", type=float, default=0.9,
        help="Минимальный порог уверенности для классификации языка (по умолчанию 0.9)"
    )

    args = parser.parse_args()

    text = read_file(args.input)
    paragraphs = split_paragraphs(text)

    langs = []
    for p in paragraphs:
        lang, conf = classify_paragraph(p, args.min_confidence)
        langs.append(lang)


    refine_short_paragraphs(paragraphs, langs, args.min_confidence)
    write_paragraphs_by_lang(paragraphs, langs, args.output_dir)


if __name__ == "__main__":
    main()
