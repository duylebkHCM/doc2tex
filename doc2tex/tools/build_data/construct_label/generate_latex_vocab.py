import sys, logging, argparse, os, pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.data_const import LABEL_KEY


def process_args(args):
    parser = argparse.ArgumentParser(description="Generate vocabulary file.")

    parser.add_argument(
        "--label-path",
        dest="label_path",
        type=str,
        required=True,
        help=("Directory of training dataframe"),
    )
    parser.add_argument(
        "--output-vocab",
        dest="output_vocab",
        type=str,
        default=None,
        help=("Output file for putting vocabulary."),
    )
    parser.add_argument(
        "--unk-threshold",
        dest="unk_threshold",
        type=int,
        default=10,
        help=(
            "If the number of occurences of a token is less than (including) the threshold, then it will be excluded from the generated vocabulary."
        ),
    )
    parser.add_argument(
        "--log-path",
        dest="log_path",
        type=str,
        default="log.txt",
        help=("Log file path, default=log.txt"),
    )
    parser.add_argument(
        "--raw-vocab",
        dest="raw",
        action="store_true",
        default=False,
        help="First vocab version to debugging",
    )
    parser.add_argument(
        "--viz-dis",
        dest="viz",
        action="store_true",
        default=False,
        help="Visualize vocab tokens distribution in diagram for debugging. Use when mode raw-vocab is True",
    )
    parameters = parser.parse_args(args)
    return parameters


def main(args):
    parameters = process_args(args)
    parent_dir = Path(parameters.label_path).parent
    label_ver = parent_dir.name
    if not parameters.output_vocab:
        vocab_path = parent_dir.parent.parent.joinpath("vocab").joinpath(label_ver)
    else:
        vocab_path = Path(parameters.output_vocab).joinpath(label_ver)
    if not vocab_path.exists():
        vocab_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s",
        filename=vocab_path.joinpath(parameters.log_path),
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s"
    )
    console.setFormatter(formatter)
    logging.getLogger("VOCAB-LOGGER").addHandler(console)

    logging.info("Script being executed: %s" % __file__)

    if vocab_path.joinpath("vocab.txt").exists() and parameters.raw and parameters.viz:
        tokens = [
            line.strip().split("\t")
            for line in open(
                str(vocab_path.joinpath("vocab.txt")), "r", encoding="utf-8"
            ).readlines()
        ]
        tokens = list(zip(*tokens))
        tokens = {k: v for k, v in zip(["tokens", "freq"], tokens)}
        token_df = pd.DataFrame(data=tokens)
        print("token_df", token_df.head(10))
        plt.hist(token_df["freq"])
        plt.xlabel("Token")
        plt.ylabel("Frequency")
        plt.savefig(str(vocab_path.joinpath("ditribution.jpg")))
        sys.exit(0)

    train_df = pd.read_csv(parameters.label_path)
    vocab = {}
    for _, row in train_df.iterrows():
        lbl = row.loc[LABEL_KEY.LABEL.value]
        tokens = lbl.split()
        tokens_out = []
        for token in tokens:
            tokens_out.append(token)
            if token not in vocab:
                vocab[token] = 0
            vocab[token] += 1

    vocab_out = []
    unknown = []
    vocab_sort = sorted(
        [(k, v) for k, v in vocab.items()], key=lambda x: x[1], reverse=True
    )

    if parameters.raw:
        for word, freq in vocab_sort:
            item = (word, freq)
            if freq > parameters.unk_threshold:
                vocab_out.append(item)
            else:
                unknown.append(item)

        vocab = [word for word in vocab_out]

        with open(str(vocab_path.joinpath("vocab.txt")), "w", encoding="utf-8") as fout:
            fout.write(
                "\n".join(["\t".join([item[0], str(item[1])]) for item in vocab])
            )
    else:
        vocab_sort = [item[0] for item in vocab_sort]
        for word in vocab_sort:
            if vocab[word] > parameters.unk_threshold:
                vocab_out.append(word)
            else:
                unknown.append((word, vocab[word]))

        vocab = sorted([word for word in vocab_out])

        with open(
            str(vocab_path.joinpath("vocab_full_filter.txt")), "w", encoding="utf-8"
        ) as fout:
            fout.write("\n".join(vocab))

    for unk in unknown:
        logging.info("#UNK's: %s with %d apperances" % (unk[0], unk[1]))


if __name__ == "__main__":
    main(sys.argv[1:])
    logging.info("Jobs finished")
