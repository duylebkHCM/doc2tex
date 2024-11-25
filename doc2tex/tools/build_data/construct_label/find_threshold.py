import pandas as pd
import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
from itertools import dropwhile

sns.set_style("darkgrid")

if __name__ == "__main__":
    vocab_path = sys.argv[1]
    vocab = open(str(vocab_path), "r", encoding="utf-8").readlines()
    vocab = [line.strip() for line in vocab]
    tokens, freq = [], []

    for item in vocab:
        item = item.split("\t")
        assert len(item) == 2
        tokens.append(item[0])
        freq.append(int(item[1]))

    data = {"token": tokens, "freq": freq}
    df = pd.DataFrame.from_dict(data)

    # calculate cumsum of freq
    df = df.sort_values(by="freq", ascending=True, axis=0)
    df["freq_cumsum"] = np.cumsum(df["freq"])
    cut_threshold = 0.001  # cut at the point where accumalated sum is less than 1% of total number of all tokens in dataset (less contribution)

    df = df.sort_values(by="freq_cumsum", ascending=True, axis=0)
    df.reset_index(drop=True, inplace=True)

    last_idx = len(df) - 1

    final_row = None
    for idx, row in df.iterrows():
        if row.loc["freq_cumsum"] >= int(
            cut_threshold * df.loc[last_idx, "freq_cumsum"]
        ):
            final_row = row
            break

    recheck_vocab = open(
        str(Path(vocab_path).parent.joinpath("recheck_vocab.txt")),
        "w",
        encoding="utf-8",
    )

    # save vocab to recheck outlier
    for item in dropwhile(lambda x: x.split("\t")[0] != final_row.loc["token"], vocab):
        recheck_vocab.write(item + "\n")

    # plot histogram of all vocab with line cut at the threshold cut. For paper visualize purpose
    freq_data = df["freq"].values
    sns.displot(freq_data, kde=True)
    plt.savefig(str(Path(vocab_path).parent.joinpath("threshold.jpg")))
