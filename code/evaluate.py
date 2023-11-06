"""Evaluate model predictions against gold standard or another model (IAA)."""

import argparse

from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from utils.helpers import read_corpus


def create_arg_parser() -> argparse.Namespace:
    """Return parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_data", type=str, help="File containing test data (gold standards)")
    parser.add_argument("-p", "--prediction_data", type=str, help="File containing model predictions")
    parser.add_argument(
        "-c",
        "--comparison_data",
        type=str,
        help="File containing other predictions to compare against",
    )
    return parser.parse_args()


def evaluate(y_test: list[str], y_pred: list[str]) -> None:
    """Evaluate predictions against gold standard."""
    print(classification_report(y_test, y_pred, digits=3))
    print(confusion_matrix(y_test, y_pred))


def main() -> None:
    """Run main function."""
    args = create_arg_parser()
    if args.test_data:
        _x, y_test = read_corpus(args.test_data)
        with open(args.prediction_data) as file:
            y_pred = [line.strip() for line in file]
        evaluate(y_test, y_pred)
    elif args.comparison_data:
        with open(args.comparison_data) as file:
            y_comp = [line.strip() for line in file]
        with open(args.prediction_data) as file:
            y_pred = [line.strip() for line in file]
        evaluate(y_comp, y_pred)
        print(f"cohen's kappa: {round(cohen_kappa_score(y_pred, y_comp)*100, 1)}%")


main()
