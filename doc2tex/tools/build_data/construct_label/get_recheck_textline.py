import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.data_const import LABEL_KEY
from tqdm import tqdm
import pickle

if __name__ == "__main__":
    total_path = sys.argv[1]
    vocab_recheck = sys.argv[2]

    total_df = pd.read_csv(total_path)

    recheck_vocab = open(vocab_recheck, "r", encoding="utf-8").readlines()
    recheck_vocab = [line.strip().split("\t") for line in recheck_vocab]
    total_df.reset_index(drop=True, inplace=True)

    if not Path(total_path).parent.joinpath("temp_storage.pkl").exists():
        indices = []
        sub_df = []
        for idx, row in tqdm(total_df.iterrows(), total=len(total_df)):
            latex_str = row.loc[LABEL_KEY.LABEL.value]
            latex_split = latex_str.strip().split()

            for item in recheck_vocab:
                tok = item[0]
                if tok in latex_split:
                    sub_df.append(row)
                    indices.append(idx)
                    break

        with open(str(Path(total_path).parent.joinpath("temp_storage.pkl")), "wb") as f:
            pickle.dump((sub_df, indices), f)
    else:
        with open(str(Path(total_path).parent.joinpath("temp_storage.pkl")), "rb") as f:
            sub_df, indices = pickle.load(f)

    recheck_freq = {}
    for item in recheck_vocab:
        tok = item[0]
        for tl in sub_df:
            latex = tl.loc[LABEL_KEY.LABEL.value].split()
            for t in latex:
                if t == tok:
                    if tok not in recheck_freq:
                        recheck_freq[tok] = 1
                    else:
                        recheck_freq[tok] += 1

    # check the freq in recheck_vocab.txt and in recheck_feq dic is the same
    for item in recheck_vocab:
        tok, freq = item
        assert recheck_freq[tok] == int(
            freq
        ), f"{tok} has {recheck_freq[tok]} in dict and {int(freq)} in txt file"

    # drop sub_df from total_df using indices
    assert len(list(set(list(total_df.index)))) == len(
        total_df
    ), f"Indices should all unique"
    print("Prev length", len(total_df))
    total_df_new = total_df[~total_df.index.isin(indices)]
    print("Truncate length", len(total_df_new))
    total_df_new.to_csv(
        str(Path(total_path).parent.joinpath("total_truncate_recheck_textline.csv")),
        index=False,
        header=True,
    )

    # combine sub_df list into new df
    concat_df = pd.concat(sub_df, axis=1)
    concat_df = concat_df.T
    concat_df.reset_index(drop=True, inplace=True)
    concat_df.to_csv(
        str(Path(total_path).parent.joinpath("recheck_df.csv")),
        index=False,
        header=True,
    )  # default to write in UTF-8 format in python 3, https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html?highlight=to_csv#pandas.DataFrame.to_csv
