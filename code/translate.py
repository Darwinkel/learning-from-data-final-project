"""Translate documents in a dataset into another language."""
import argparse

import pandas as pd
from transformers import pipeline


def create_arg_parser() -> argparse.Namespace:
    """Return parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-file",
        default="data/train.tsv",
        type=str,
        help="Input file to translate (default data/train.tsv)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default="data/train-nl.tsv",
        type=str,
        help="Name of the output file (default data/train-nl.tsv)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="nl",
        type=str,
        help="Language code of the target language supported by Helsinki-NLP/opus-mt-en- (default nl)",
    )

    return parser.parse_args()


def data(documents: list[str]) -> str:
    """Yield documents from a list."""
    yield from documents


def main() -> None:
    """Run main function."""
    args = create_arg_parser()

    dataframe = pd.read_csv(args.input_file, sep="\t", header=None)
    documents = dataframe.iloc[:, 0].to_list()
    labels = dataframe.iloc[:, 1].to_list()

    translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{args.language}")

    translations = translator(documents)

    with open(args.output_file, "w") as file:
        for idx, translation in enumerate(translations):
            print(idx)
            print(translation)
            file.write(f"{translation['translation_text']}\t{labels[idx]}\n")


main()
