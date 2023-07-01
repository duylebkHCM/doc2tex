import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.data_const import LABEL_KEY
import json
from collections import Counter
import numpy as np
import tqdm

if __name__ == "__main__":
    np.random.seed(1999)

    total_path = sys.argv[1]
    if len(sys.argv) < 3:
        debug = 0
    else:
        debug = int(sys.argv[2])

    total_df = pd.read_csv(total_path, encoding="utf-8")
    total_df.reset_index(drop=True, inplace=True)

    if debug:
        # review katex supported tokens
        total_tokens = 0
        info = {}
        with open("katex_support_latex_token.json", "w", encoding="utf-8") as f:
            for token_group_path in Path("katex_tokens").glob("*.txt"):
                info[token_group_path.stem] = []
                group_tokens = open(
                    str(token_group_path), "r", encoding="utf-8"
                ).readlines()
                group_tokens = [line.strip().split() for line in group_tokens]
                total_tokens += sum([len(line) for line in group_tokens])
                for tok in group_tokens:
                    info[token_group_path.stem].append("\t".join(tok))
            print("Total katex tokens", total_tokens)
            json.dump(info, f, indent=4, ensure_ascii=False)
    else:
        assert Path("katex_support_latex_token.json").exists()
        with open("katex_support_latex_token.json", "r", encoding="utf-8") as f:
            token_lsts = json.load(f)

        # if not Path(recheck_path).parent.joinpath('filtered_oov')
        remove_indcies = []
        # standardie other tokens not in recheck_vocab
        for idx, row in tqdm.tqdm(total_df.iterrows(), total=len(total_df)):
            c = Counter()
            for group_tok in token_lsts:
                for inner_tok in token_lsts[group_tok]:
                    inner_tok = inner_tok.split("\t")
                    row_toks = row.loc[LABEL_KEY.LABEL.value].split()
                    for tok in row_toks:
                        if (tok in inner_tok) and (tok != inner_tok[0]):
                            # standard
                            if group_tok not in ["font", "op"]:
                                row.loc[LABEL_KEY.LABEL.value] = row.loc[
                                    LABEL_KEY.LABEL.value
                                ].replace(tok, inner_tok[0])
                            elif group_tok == "op":
                                if tok == "\\operatornamewithlimits":
                                    row.loc[LABEL_KEY.LABEL.value] = row.loc[
                                        LABEL_KEY.LABEL.value
                                    ].replace(tok, inner_tok[0])
                                else:
                                    new_tok = (
                                        inner_tok[0]
                                        + " { "
                                        + " ".join([i for i in tok[1:]])
                                        + " }"
                                    )
                                    row.loc[LABEL_KEY.LABEL.value] = row.loc[
                                        LABEL_KEY.LABEL.value
                                    ].replace(tok, new_tok)
                            elif group_tok == "font":
                                if not tok.startswith("\\math") and not tok.startswith(
                                    "\\text"
                                ):
                                    new_tok = inner_tok[0] + " {"
                                    old_tok1 = "{ " + tok
                                    old_tok2 = tok + " {"
                                    row.loc[LABEL_KEY.LABEL.value] = row.loc[
                                        LABEL_KEY.LABEL.value
                                    ].replace(old_tok1, new_tok)
                                    row.loc[LABEL_KEY.LABEL.value] = row.loc[
                                        LABEL_KEY.LABEL.value
                                    ].replace(old_tok2, new_tok)
                                else:
                                    row.loc[LABEL_KEY.LABEL.value] = row.loc[
                                        LABEL_KEY.LABEL.value
                                    ].replace(tok, inner_tok[0])

                            c.update(row.loc[LABEL_KEY.LABEL.value].split())
                            freq = dict(c.most_common()).get(tok, 0)
                            freq_new = dict(c.most_common()).get(inner_tok[0], 0)
                            if freq > 0 or freq_new == 0:
                                remove_indcies.append(idx)
                                break
                    else:
                        continue
                    break
                else:
                    continue
                break

            del c

        total_processed_df = total_df[~total_df.index.isin(remove_indcies)]
        total_processed_df.reset_index(drop=True, inplace=True)
        total_processed_df.to_csv(
            str(Path(total_path).parent.joinpath("total_filtered.csv")),
            header=True,
            index=False,
            encoding="utf-8",
        )
