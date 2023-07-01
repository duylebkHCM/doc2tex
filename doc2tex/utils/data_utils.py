import numpy as np
from PIL import Image
import cv2
import re
from collections import deque
import math
from typing import List


def pad(img: Image, divable=32):
    """Pad an Image to the next full divisible value of `divable`. Also normalizes the PIL.image and invert if needed.

    Args:
        img (PIL.Image): input PIL.image
        divable (int, optional): . Defaults to 32.

    Returns:
        PIL.Image
    """
    data = np.array(img.convert("LA"))

    data = (data - data.min()) / (data.max() - data.min()) * 255
    if data[..., 0].mean() > 128:
        gray = 255 * (data[..., 0] < 128).astype(
            np.uint8
        )  # To invert the text to white
    else:
        gray = 255 * (data[..., 0] > 128).astype(np.uint8)
        data[..., 0] = 255 - data[..., 0]

    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = data[b : b + h, a : a + w]

    if rect[..., -1].var() == 0:
        im = Image.fromarray((rect[..., 0]).astype(np.uint8)).convert("L")
    else:
        im = Image.fromarray((255 - rect[..., -1]).astype(np.uint8)).convert("L")
    dims = []

    for x in [w, h]:
        div, mod = divmod(x, divable)
        dims.append(divable * (div + (1 if mod > 0 else 0)))

    padded = Image.new("L", dims)
    padded.paste(im, im.getbbox())
    return padded


def get_divisible_size(ori_h, ori_w, max_dimension=None, scale_factor=32):
    if ori_h % scale_factor:
        new_h = math.ceil(ori_h / scale_factor) * scale_factor
        if new_h > max_dimension[0]:
            new_h = math.floor(ori_h / scale_factor) * scale_factor
    if ori_w % scale_factor:
        new_w = math.ceil(ori_w / scale_factor) * scale_factor
        if new_w > max_dimension[1]:
            new_w = math.floor(ori_w / scale_factor) * scale_factor
    return int(new_h), int(new_w)


def minmax_size(img, max_dimensions=None, min_dimensions=None, is_gray=True):
    if max_dimensions is not None:
        ratios = [a / b for a, b in zip(list(img.size)[::-1], max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size) / max(ratios)
            new_h, new_w = get_divisible_size(size[1], size[0], max_dimensions)
            img = img.resize((new_w, new_h), Image.LANCZOS)

    if min_dimensions is not None:
        ratios = [a / b for a, b in zip(list(img.size)[::-1], min_dimensions)]
        if any([r < 1 for r in ratios]):
            new_h, new_w = img.size[1] / min(ratios), img.size[0] / min(ratios)
            new_h, new_w = get_divisible_size(new_h, new_w, max_dimensions)
            if is_gray:
                MODE = "L"
                BACKGROUND = 255
            padded_im = Image.new(MODE, (new_w, new_h), BACKGROUND)
            padded_im.paste(img, img.getbbox())
            img = padded_im

    return img


def get_size(ori_w, ori_h, config):
    min_dim, max_dim = config["min_dimension"], config["max_dimension"]
    if config["downsample"]:
        ori_h, ori_w = int(ori_h / config["downsample"]), int(
            ori_w / config["downsample"]
        )

    scale_factor = config["scale_factor"]
    new_h, new_w = get_divisible_size(ori_h, ori_w, scale_factor)

    if any(
        [
            dim % scale_factor != 0
            for limit_size in (min_dim, max_dim)
            for dim in limit_size
        ]
    ):
        raise ValueError("Min max dimension should divisible by scale factor")

    ratios = [a / b for a, b in zip((new_h, new_w), tuple(max_dim))]
    if any([r > 1 for r in ratios]):
        new_h, new_w = new_h / max(ratios), new_w / max(ratios)
        new_h, new_w = get_divisible_size(new_h, new_w, max_dim, scale_factor)

    ratios = [a / b for a, b in zip((new_h, new_w), tuple(min_dim))]
    if any([r < 1 for r in ratios]):
        new_h, new_w = new_h / min(ratios), new_w / min(ratios)
        new_h, new_w = get_divisible_size(new_h, new_w, scale_factor)

    return new_h, new_w


def remove_reduntant_bracket(lst_tokens: List[str]):
    switch = 0
    left_idx_lst = []
    final = []

    for idx, tok in enumerate(lst_tokens):
        if tok == "{":
            switch += 1
            left_idx_lst.append(idx)
        else:
            if tok == "}" and switch > 0:
                switch -= 1
                final.append(left_idx_lst.pop(-1))
                final.append(idx)
            else:
                switch = 0
                left_idx_lst = []

    assert not divmod(len(final), 2)[-1]
    if len(final) > 1:
        new_lst_tokens = [tok for idx, tok in enumerate(lst_tokens) if idx not in final]
        return " ".join(new_lst_tokens)
    else:
        return " ".join(lst_tokens)


def standardize_whitespace_v2(latex_str, std_ws, standard_dict):
    ws_ptn = re.compile(r"(\\%s\s)+" % std_ws)
    matchers = re.finditer(ws_ptn, latex_str)

    if matchers:
        standard_tokens = ""
        prev_end_idx = 0
        for matcher in matchers:
            start_idx, end_idx = matcher.start(0), matcher.end(0)
            standard_tokens += latex_str[prev_end_idx:start_idx]
            ws_group = latex_str[start_idx : end_idx - 1]
            ws_group = ws_group.split()

            if len(ws_group) > 1:
                for space_len in standard_dict:
                    if len(ws_group) < space_len:
                        ws_group = [std_ws] * standard_dict[space_len]
                        break
                else:
                    ws_group = [std_ws] * list(standard_dict.keys())[-1]

            standard_tokens += " ".join(ws_group)
            standard_tokens += " "
            prev_end_idx = end_idx

        standard_tokens += latex_str[prev_end_idx:]

        return standard_tokens
    else:
        return latex_str


def remove_redundant_array_env(text: str):
    raw_text = str(text)
    text = text.strip().split()
    begin_storage = []
    scope_storage = []

    # collect all group of array environments
    for idx, t in enumerate(text):
        if t == r"\begin{array}":
            begin_storage.append(idx)
        elif t == r"\end{array}":
            scope = [begin_storage.pop(-1), idx + 1]
            scope_storage.append(scope)

    # sort based on length of each group. Processed shortest a.k.a inner most group first.
    distance_lst = [(item[1] - item[0]) for item in scope_storage]
    # waiting for processing
    process_order = sorted(list(zip(scope_storage, distance_lst)), key=lambda x: x[1])

    # already processed groups storage
    processed = []
    while len(process_order):
        # get first item in the list
        processed_group_info = process_order[0]
        start_idx = processed_group_info[0][0]
        end_idx = processed_group_info[0][1]
        merge_lsts = []

        if len(processed):
            sort_processed = sorted(processed, key=lambda x: x[0][0], reverse=False)
            for idx, group in enumerate(sort_processed):
                group_start = group[0][0]
                group_end = group[0][1]
                if group_start > start_idx and group_end < end_idx:
                    merge_lsts.append(idx)
        else:
            sort_processed = []

        if merge_lsts:
            processed_group = " ".join(text[start_idx:end_idx])
            # ensure there are no string with such pattern
            precheck_pttn = re.compile(r"\d+\,\s\d+")
            if re.findall(precheck_pttn, processed_group) != []:
                raise ValueError("Find an exception with not follow current logic.")
            # encode already processed child group
            for idx in merge_lsts:
                processed_group = processed_group.replace(
                    " ".join(
                        text[sort_processed[idx][0][0] : sort_processed[idx][0][1]]
                    ),
                    f"{sort_processed[idx][0][0]}, {sort_processed[idx][0][1]}",
                )
            processed_group = processed_group.split()
        else:
            processed_group = text[start_idx:end_idx]

        try:
            extract_lsts = []
            for tok in processed_group:
                if tok == "\\\\":
                    # keep it the same
                    extract_lsts = processed_group
                    break
            else:
                ptn_checked = re.compile(
                    r"\\begin{array}\s\{[rlc\s]+\}\s(\&?\s?\{.*\})+\s\\end{array}"
                )
                concat_group = " ".join(processed_group)
                matcher = re.match(ptn_checked, concat_group)
                if matcher:
                    extract_comp_ptn = re.compile(r"\{([^\&]+)\}")
                    body = concat_group[matcher.start(1) : matcher.end(1)]
                    comp_matcher = re.finditer(extract_comp_ptn, body)
                    for m in comp_matcher:
                        extract_lsts.append(body[m.start(1) : m.end(1)])
                    padding_whitespace = len(processed_group) - len(extract_lsts)
                    extract_lsts += [" "] * padding_whitespace
                    assert len(extract_lsts) == len(processed_group)
                else:
                    raise ValueError("There is something wrong")
        except ValueError:
            extract_lsts = processed_group

        # complete checking process, decode previously encoded sub-strings
        extract_lsts = " ".join(extract_lsts)
        if merge_lsts:
            for item in merge_lsts:
                assert len(sort_processed[item]) == 2
                replace_info = sort_processed[item][1]
                signal = sort_processed[item][0]
                str_signal = ", ".join(list(map(str, signal)))
                extract_lsts = extract_lsts.replace(str_signal, replace_info)

        info = ([start_idx, end_idx], extract_lsts)
        # clear previous processed lists which are children of current processed group
        sort_processed = (
            [item for idx, item in enumerate(sort_processed) if idx not in merge_lsts]
            if merge_lsts
            else sort_processed
        )
        processed = [info] + sort_processed
        # remove processed group from waiting list
        process_order.pop(0)

    # Relace raw text with all processed group and return
    for new_group in processed:
        group_start = new_group[0][0]
        group_end = new_group[0][1]
        replace_text = raw_text.split()[group_start:group_end]
        replace_text = " ".join(replace_text)
        raw_text = raw_text.replace(replace_text, new_group[1])

    remove_redundant_whitespace_text = " ".join(raw_text.split())

    return remove_redundant_whitespace_text


class Postprocessing:
    # left bracket patterns
    re_angle_left = re.compile(r"\\langle", flags=re.DOTALL)
    re_parens_open = re.compile(r"\(", flags=re.DOTALL)
    re_parens_left = re.compile(r"\\left\(", flags=re.DOTALL)
    re_braces_open = re.compile(r"\{", flags=re.DOTALL)
    re_braces_left = re.compile(r"\\left\\\{", flags=re.DOTALL)
    re_square_open = re.compile(r"\[", flags=re.DOTALL)
    re_square_left = re.compile(r"\\left\[", flags=re.DOTALL)
    # right bracket patterns
    re_angle_right = re.compile(r"\\rangle", flags=re.DOTALL)
    re_parens_close = re.compile(r"\)", flags=re.DOTALL)
    re_parens_right = re.compile(r"\\right\)", flags=re.DOTALL)
    re_braces_close = re.compile(r"\}", flags=re.DOTALL)
    re_braces_right = re.compile(r"\\right\\\}", flags=re.DOTALL)
    re_square_close = re.compile(r"\]", flags=re.DOTALL)
    re_square_right = re.compile(r"\\right\]", flags=re.DOTALL)

    @staticmethod
    def replace_brackets(string, pattern: re.Pattern, sub_pattern: re.Pattern):
        string = re.sub(pattern, sub_pattern.pattern.replace("\\", ""), string)
        return string

    @staticmethod
    def run_stack(
        string, re_either: re.Pattern, re_left: re.Pattern, re_right: re.Pattern
    ):
        matches = re.finditer(re_either, string)
        match_info = [(m.group(), m.start(0)) for m in matches]
        match_list = deque(match_info)

        if len(match_list) == 0:
            return string  # early exit if no brackets => no balancing needed

        balance_stack = deque()
        for item in match_list:
            if re_left.match(item[0]):
                current_bracket = ("l", item[1])
            elif re_right.match(item[0]):
                current_bracket = ("r", item[1])
            else:
                raise ValueError(f"got problematic bracket '{item[0]}' in 'balance'")
            previous_bracket = balance_stack[-1] if len(balance_stack) > 0 else None

            if (
                previous_bracket is not None
                and (previous_bracket[0] == "l")
                and (current_bracket[0] == "r")
            ):
                balance_stack.pop()
            else:
                balance_stack.append(current_bracket)

        return balance_stack

    @staticmethod
    def balance(
        string: str,
        re_left: re.Pattern,
        re_right: re.Pattern,
    ) -> str:
        """
        for a given bracket type, identify all occurrences of the current bracket,
            and balance them using the standard stack-based algorithm; Python collections'
            'deque' data structure serves the purpose of a stack here.
        """
        re_either = re.compile(
            re_left.pattern + "|" + re_right.pattern, flags=re.DOTALL
        )

        balance_stack = Postprocessing.run_stack(string, re_either, re_left, re_right)

        if isinstance(balance_stack, str):
            return balance_stack

        # whatever's left on the stack is the imbalance
        imbalance_right = [item for item in balance_stack if item[0] == "r"]
        imbalance_right = sorted(imbalance_right, key=lambda x: x[1], reverse=False)

        if imbalance_right:
            for idx, item in enumerate(imbalance_right):
                insert_idx = item[1]
                insert_idx += idx
                string = (
                    string[:insert_idx]
                    + re_left.pattern.replace("\\", "")
                    + string[insert_idx:]
                )

        balance_stack = Postprocessing.run_stack(string, re_either, re_left, re_right)

        imbalance_left = [item for item in balance_stack if item[0] == "l"]
        imbalance_left = sorted(imbalance_left, key=lambda x: x[1], reverse=False)

        if imbalance_left:
            for idx, item in enumerate(imbalance_left):
                insert_idx = item[1]
                if idx > 0:
                    insert_idx += idx
                string = (
                    string[: insert_idx + 1]
                    + re_right.pattern.replace("\\", "")
                    + string[insert_idx + 1 :]
                )

        return string

    # main loop
    @staticmethod
    def pipeline(snippet):
        snippet = snippet.strip()
        result = Postprocessing.replace_brackets(
            snippet, Postprocessing.re_parens_left, Postprocessing.re_parens_open
        )
        result = Postprocessing.replace_brackets(
            result, Postprocessing.re_braces_left, Postprocessing.re_braces_open
        )
        result = Postprocessing.replace_brackets(
            result, Postprocessing.re_square_left, Postprocessing.re_braces_open
        )
        result = Postprocessing.replace_brackets(
            result, Postprocessing.re_braces_right, Postprocessing.re_braces_close
        )
        result = Postprocessing.replace_brackets(
            result, Postprocessing.re_square_right, Postprocessing.re_square_close
        )
        result = Postprocessing.replace_brackets(
            result, Postprocessing.re_parens_right, Postprocessing.re_parens_close
        )
        result = Postprocessing.balance(
            result, Postprocessing.re_parens_open, Postprocessing.re_parens_close
        )
        result = Postprocessing.balance(
            result, Postprocessing.re_braces_open, Postprocessing.re_braces_close
        )
        result = Postprocessing.balance(
            result, Postprocessing.re_square_open, Postprocessing.re_square_close
        )
        return result

    @staticmethod
    def remove_unused_whitespace(s: str):
        """Remove unnecessary whitespace from LaTeX code.

        Args:
            s (str): Input string

        Returns:
            str: Processed PIL.image
        """
        text_reg = r"(\\(operatorname|mathrm|mathbf|mathsf|mathit|mathfrak|mathnormal)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = "[\W_^\d]"
        names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s
