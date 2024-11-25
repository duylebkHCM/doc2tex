import argparse
import os
import re
import numpy as np
from typing import List
from standard_const import *


def check_brackets(s):
    a = []
    surrounding = False
    for i, c in enumerate(s):
        if c == "{":
            if i > 0 and s[i - 1] == "\\":  # not perfect
                continue
            else:
                a.append(1)
            if i == 0:
                surrounding = True
        elif c == "}":
            if i > 0 and s[i - 1] == "\\":
                continue
            else:
                a.append(-1)

    b = np.cumsum(a)

    if len(b) > 1 and b[-1] != 0:
        raise ValueError(s)

    surrounding = s[-1] == "}" and surrounding
    if not surrounding:
        return s
    elif (b == 0).sum() == 1:
        return s[1:-1]
    else:
        return s


def remove_labels(string):
    for s in LABEL_TAGS:
        string = re.sub(s, "", string)
    return string


def clean_matches(matches, min_chars=MIN_CHARS):
    faulty = []
    graphic_token = [
        "tikz",
        r"\begin{picture}",
        r"\begin{fmfgraph}",
        r"\bigcirc",
        r"\bigotimes",
        r"\color",
    ]

    for i in range(len(matches)):
        if any([tok in matches[i] for tok in graphic_token]):
            faulty.append(i)
            continue

        matches[i] = remove_labels(matches[i])
        matches[i] = (
            matches[i]
            .replace("\n", "")
            .replace(r"\notag", "")
            .replace(r"\nonumber", "")
        )

        # matches[i] = re.sub(r'\\noalign(.*)', '', matches[i])
        matches[i] = re.sub(OUTER_WHITESPACE, "", matches[i])

        if len(matches[i]) < min_chars:
            faulty.append(i)
            continue

        try:
            matches[i] = check_brackets(matches[i])
        except ValueError:
            faulty.append(i)

        if matches[i][-1] == "\\" or "newcommand" in matches[i][-1]:
            faulty.append(i)

    matches = [m.strip() for i, m in enumerate(matches) if i not in faulty]
    return list(set(matches))


def find_math(s: str) -> List[str]:
    r"""Find all occurences of math in a Latex-like document.

    Args:
        s (str): String to search
        wiki (bool, optional): Search for `\displaystyle` as it can be found in the wikipedia page source code. Defaults to False.

    Returns:
        List[str]: List of all found mathematical expressions
    """
    matches = []
    patterns = [DOLLAR, EQUATION, ALIGN]
    groups = [1, 1, 0]

    for i, pattern in zip(groups, patterns):
        x = re.findall(pattern, s)
        matches.extend([g[i] for g in x])

    return clean_matches(matches)
