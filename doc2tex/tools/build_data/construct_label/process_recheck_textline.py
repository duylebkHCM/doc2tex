import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.data_const import LABEL_KEY
import json
from collections import defaultdict, namedtuple, Counter
import numpy as np
import enum
from typing import List
import pickle


class TL_Flag(enum.Enum):
    EXISTED = 0
    UNADDED = 1
    ADDED = 2


Textline = namedtuple("Textline", "idx flag freq")

if __name__ == "__main__":
    np.random.seed(1999)

    recheck_path = sys.argv[1]
    recheck_vocab_path = sys.argv[2]

    recheck_vocab = open(recheck_vocab_path, "r", encoding="utf-8").readlines()
    recheck_vocab = [line.strip().split("\t") for line in recheck_vocab]
    recheck_vocab = sorted(recheck_vocab, key=lambda x: int(x[1]), reverse=True)

    recheck_df = pd.read_csv(recheck_path)
    recheck_df.reset_index(drop=True, inplace=True)

    assert Path("katex_support_latex_token.json").exists()
    with open("katex_support_latex_token.json", "r", encoding="utf-8") as f:
        token_lsts = json.load(f)

    upsampling_num = int(recheck_vocab[0][1])

    if (
        not Path(recheck_path).parent.joinpath("contained_indices.pkl").exists()
        and not Path(recheck_path).parent.joinpath("unresetindex_dupp_df.csv").exists()
    ):
        remove_indcies = []
        remove_toks = []
        # standardize again -_-
        contained_indices = defaultdict(list)
        for _, (tok, _) in enumerate(recheck_vocab):
            for group_tok in token_lsts:
                for inner_tok in token_lsts[group_tok]:
                    inner_tok = inner_tok.split("\t")
                    # exception group: op, font
                    if tok in inner_tok:
                        assert (
                            tok == inner_tok[0]
                        ), f"Token: {tok}, while inner_tok[0]: {inner_tok[0]}"
                        for idx, row in recheck_df.iterrows():
                            if idx in remove_indcies:
                                continue
                            if tok in row.loc[LABEL_KEY.LABEL.value].split():
                                c = Counter()
                                c.update(row.loc[LABEL_KEY.LABEL.value].split())
                                freq = dict(c.most_common()).get(tok, 0)
                                assert freq > 0
                                tl_info = Textline(
                                    idx=idx, flag=TL_Flag.EXISTED.value, freq=freq
                                )
                                prev_version = list(
                                    filter(
                                        lambda x: x.idx == idx, contained_indices[tok]
                                    )
                                )
                                if not len(prev_version):
                                    contained_indices[tok].append(
                                        tl_info
                                    )  # default assign all tl_idx in dict with 0 flag indicate it exist in df.
                        break
                else:
                    continue
                break
            else:
                # token do not exists in general vocab, remove all textline contain tokens
                remove_toks.append(tok)
                for idx, row in recheck_df.iterrows():
                    if tok in row.loc[LABEL_KEY.LABEL.value].split():
                        if idx not in remove_indcies:
                            remove_indcies.append(idx)

        remove_indcies = list(set(remove_indcies))  # remoe dupplicate indices
        processed_df = recheck_df[~recheck_df.index.isin(remove_indcies)]

        # duplicate to reach upsampling_num, create new final df
        dupp_df = processed_df.copy()
        # save contained_indices
        with open(
            str(Path(recheck_path).parent.joinpath("contained_indices.pkl")), "wb"
        ) as f:
            pickle.dump((contained_indices, remove_indcies), f)
        dupp_df.to_csv(
            str(Path(recheck_path).parent.joinpath("unresetindex_dupp_df.csv"))
        )

        processed_df.reset_index(drop=True, inplace=True)
        processed_df.to_csv(
            str(Path(recheck_path).parent.joinpath("recheck_df_processed.csv")),
            index=False,
            header=True,
        )
    else:
        with open(
            str(Path(recheck_path).parent.joinpath("contained_indices.pkl")), "rb"
        ) as f:
            contained_indices, remove_indcies = pickle.load(f)
        dupp_df = pd.read_csv(
            str(Path(recheck_path).parent.joinpath("unresetindex_dupp_df.csv"))
        )
        dupp_df.set_index(keys="Unnamed: 0", inplace=True)
        dupp_df.index.name = None

    # update contained_indices iteratively
    for tok in contained_indices:
        contained_indices[tok] = [
            item for item in contained_indices[tok] if item.idx not in remove_indcies
        ]

    contained_indices = dict(
        [item for item in contained_indices.items() if len(item[1]) > 0]
    )

    for tok in contained_indices:
        lst_tl: List[Textline] = contained_indices.get(tok, None)
        assert lst_tl is not None
        if not len(lst_tl):
            print(f"{tok} tokens has no textlines")
            continue

        num_existed_tok = sum([item.freq for item in lst_tl])
        num_dupplicate = upsampling_num - num_existed_tok

        if num_dupplicate <= 0:  # Reach threshold point
            continue

        lst_idx = [item.idx for item in lst_tl]  # get tl_idx only
        dupp_lst: np.ndarray = np.random.choice(
            lst_idx, size=num_dupplicate, replace=True
        )
        dupp_lst = dupp_lst.tolist()
        dupp_lst: List[Textline] = [
            Textline(
                idx=idx,
                flag=TL_Flag.ADDED.value,  # default assign all tl_idx in dict with 1 flag indicate it do not exist in df and need to add to.
                freq=list(filter(lambda x: x.idx == idx, lst_tl))[0].freq,
            )
            for idx in dupp_lst
        ]

        # append dupp_lst to lst_tl
        lst_tl += dupp_lst
        contained_indices[tok] = lst_tl

        for inner_tok in contained_indices:
            if inner_tok == tok:
                continue

            for tl in dupp_lst:
                if tl.idx in contained_indices[inner_tok]:
                    if (
                        sum([item.freq for item in contained_indices[inner_tok]])
                        < upsampling_num
                    ):
                        # default assign all tl_idx in dict with 2 flag indicate do not need to add.
                        contained_indices[inner_tok].append(
                            Textline(
                                idx=tl.idx, flag=TL_Flag.UNADDED.value, freq=tl.freq
                            )
                        )

    # extract textline from idx and add to dupp_df
    columns = dupp_df.columns
    add_textlines = {columns[0]: [], columns[1]: []}

    num_existed = [tl.idx]
    for tok in contained_indices:
        processed_img = defaultdict(int)
        for tl in contained_indices[tok]:
            if tl.flag in [TL_Flag.EXISTED.value, TL_Flag.UNADDED.value]:
                continue
            elif tl.flag == TL_Flag.ADDED.value:
                img_name = dupp_df.loc[
                    [tl.idx], LABEL_KEY.IMAGE_ID.value
                ].values.tolist()
                img_name = img_name[0]
                if img_name not in processed_img:
                    processed_img[img_name] = 1
                else:
                    processed_img[img_name] += 1
                img_name = (
                    img_name[:-4] + "_dupp_" + str(processed_img[img_name]) + ".png"
                )
                tl = dupp_df.loc[[tl.idx], LABEL_KEY.LABEL.value].values.tolist()
                tl = tl[0]
                add_textlines[columns[0]].append(img_name)
                add_textlines[columns[1]].append(tl)

    concat_df = pd.DataFrame(data=add_textlines)
    dupp_df: pd.DataFrame = pd.concat([dupp_df, concat_df], axis=0)
    dupp_df = dupp_df.sample(frac=1.0, random_state=1999)
    dupp_df.reset_index(drop=True, inplace=True)
    dupp_df.to_csv(
        str(Path(recheck_path).parent.joinpath("dupplicate_df.csv")),
        index=False,
        header=True,
    )

    print("Dupplicate samples df length", len(dupp_df))
    print("Compare to recheck_df", len(dupp_df) - len(recheck_df))

    # check all toekn in recheck_vocab reach num_upsampling
    recheck_freq = {}
    for tok in contained_indices:
        for _, row in dupp_df.iterrows():
            latex = row.loc[LABEL_KEY.LABEL.value].split()
            for t in latex:
                if t == tok:
                    if tok not in recheck_freq:
                        recheck_freq[tok] = 1
                    else:
                        recheck_freq[tok] += 1

    with open(
        str(Path(recheck_path).parent.joinpath("dupp_vocab_final.txt")),
        "w",
        encoding="utf-8",
    ) as f:
        for tok in recheck_freq:
            assert recheck_freq[tok] >= upsampling_num, f"{tok} has {recheck_freq[tok]}"
            f.write(tok + "\t" + str(recheck_freq[tok]) + "\n")
