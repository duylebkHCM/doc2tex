import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.data_const import LABEL_KEY
import json
import numpy as np
import enum
from typing import List
import pickle
import random
import string
from collections import defaultdict

if __name__ == "__main__":
    np.random.seed(1999)
    if len(sys.argv) > 1:
        unrenderd_img = open(sys.argv[2], "r").readlines()
        unrenderd_img = [line.strip() for line in unrenderd_img]
    else:
        unrenderd_img = None
    recheck_path = sys.argv[1]

    dupp_df = pd.read_csv(str(Path(recheck_path).joinpath("real_dupp.csv")))

    with open("katex_support_latex_token.json", "r", encoding="utf-8") as f:
        token_lsts = json.load(f)

    # Support augmentation tokens
    """
        alphabet english lower upper
        digits
        operatorname

    """

    digit_ = string.digits.replace("0", "")
    digit_ = list(digit_)
    lower_ = list(string.ascii_lowercase)
    upper_ = list(string.ascii_uppercase)
    operatorname = [
        item.split("\t")[1][1:]
        for item in token_lsts["op"]
        if len(item.split("\t")) > 1 and item.split("\t")[0] == "\\operatorname"
    ]
    aug_df = dupp_df.copy()
    count = 0
    if unrenderd_img is not None and len(unrenderd_img) < len(dupp_df):
        new_df = defaultdict(list)

    processed = []
    for img in unrenderd_img:
        if img in processed:
            continue

        lbl = dupp_df.loc[dupp_df["id"] == img, "label"].values.tolist()
        lbl = lbl[0]
        lbl_lst = lbl.strip().split()
        new_lbl = []
        pin = False
        unchange = False
        unpin = False
        op_name = []

        for idx, c in enumerate(lbl_lst):
            if not pin and not unchange:
                if not unpin:
                    if c in digit_:
                        remain_digits = [o_c for o_c in digit_ if o_c != c]
                        new_c = random.choice(remain_digits)
                        new_lbl += [new_c]
                    elif c in lower_:
                        remain_digits = [o_c for o_c in lower_ if o_c != c]
                        new_c = random.choice(remain_digits)
                        new_lbl += [new_c]
                    elif c in upper_:
                        remain_digits = [o_c for o_c in upper_ if o_c != c]
                        new_c = random.choice(remain_digits)
                        new_lbl += [new_c]
                    elif c == "\\operatorname":
                        if lbl_lst[idx + 1] == "{":
                            pin = True
                        new_lbl += [c]
                    elif c == "\\begin{array}":
                        unchange = True
                        new_lbl += [c]
                    else:
                        new_lbl += [c]
                else:
                    op_name = "".join(op_name)
                    remain_digits = [o_c for o_c in operatorname if o_c != op_name]
                    op_name = []
                    new_op_name = random.choice(remain_digits)
                    new_op_name = list(new_op_name)
                    new_lbl += new_op_name
                    new_lbl += ["}", c]
                    unpin = False
            else:
                if pin:
                    if c == "{":
                        new_lbl += [c]
                    elif c == "}":
                        pin = False
                        unpin = True
                    else:
                        op_name.append(c)
                if unchange:
                    if c == "}":
                        unchange = False
                    new_lbl += [c]

        if unrenderd_img and len(unrenderd_img) < len(dupp_df):
            new_df["id"].append(img)
            new_df["label"].append(" ".join(new_lbl))
            processed.append(img)

    # combine with total_truncate_recheck_df
    if unrenderd_img and len(unrenderd_img) < len(dupp_df):
        new_df = pd.DataFrame(new_df).to_csv(
            str(Path(recheck_path).joinpath("remain_unrendered_imgs.csv")),
            index=False,
            header=True,
        )
