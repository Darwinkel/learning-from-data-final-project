"""Calculate and show feature importances for a trained model."""

import argparse

from utils.models import load_model


def create_arg_parser() -> argparse.Namespace:
    """Return parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_file", type=str, help="Name of file containing trained model")
    return parser.parse_args()


def main() -> None:
    """Run main function."""
    args = create_arg_parser()
    model = load_model(args.model_file)
    model.interpret()


main()
