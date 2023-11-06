"""Predict labels for a given dataset using a trained model."""

import argparse

from utils.helpers import read_corpus
from utils.models import load_model


def create_arg_parser() -> argparse.Namespace:
    """Return parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, help="Input file containing data")
    parser.add_argument("-o", "--output_file", type=str, help="Name of output file")
    parser.add_argument("-m", "--model_file", type=str, help="Name of file containing a trained model")
    parser.add_argument(
        "-t",
        "--model_type",
        type=str,
        choices=["lstm", "classic", "lm"],
        help="Type of the trained model (lstm, classic, plm",
    )
    return parser.parse_args()


def main() -> None:
    """Run main function."""
    args = create_arg_parser()
    X, _y = read_corpus(args.input_file)
    model = load_model(args.model_file)
    y_pred = model.predict(X)
    with open(args.output_file, "w") as file:
        file.write("\n".join(y_pred))


main()
