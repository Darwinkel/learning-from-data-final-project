"""Preprocess data for GloVe embeddings."""

import argparse
import re

from utils.helpers import read_corpus


def create_arg_parser() -> argparse.Namespace:
    """Return parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, help="Input file to preprocess")
    parser.add_argument("-o", "--output_file", type=str, help="Name of output file")
    parser.add_argument("--glove", action="store_true", help="Apply GloVe preprocessing")
    return parser.parse_args()


def split_hashtag(m: re.Match) -> str:
    """Split hashtags into words."""
    body = m[0][1:]
    if body.upper() == body:
        return f"<HASHTAG> {body} <ALLCAPS>"
    return f"<HASHTAG>{' '.join(re.split(r'(?=[A-Z])', body))}"


def glove_pre(text: str) -> str:
    """Apply GloVe preprocessing to a single string."""
    replacements = [
        ("URL", "<URL>"),
        ("/", " / "),
        ("@USER", "<USER>"),
        (r"#\S+", split_hashtag),
        (r"[-+]?[.\d]*[\d]+[:,.\d]*", " <NUMBER> "),
        (r"([!?.]){2,}", lambda m: f"{m[1]} <REPEAT>"),
        ("  ", " "),
    ]
    preprocessed = text
    for r in replacements:
        preprocessed = re.sub(r[0], r[1], preprocessed)
    return preprocessed


def preprocess(data: list[str]) -> list[str]:
    """Apply preprocessing to textual data."""
    return [glove_pre(text) for text in data]


def main() -> None:
    """Run main function."""
    args = create_arg_parser()
    texts, labels = read_corpus(args.input_file)
    preprocessed_texts = preprocess(texts) if args.glove else texts
    preprocessed_data = ["\t".join(x) for x in zip(preprocessed_texts, labels)]
    with open(args.output_file, "w") as file:
        file.write("\n".join(preprocessed_data))


main()
