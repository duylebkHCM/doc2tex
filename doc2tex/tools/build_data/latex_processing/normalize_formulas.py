import sys
import os
import re
import argparse
import logging
import subprocess
import shutil
from standard_const import *
import sys

sys.path.append("../../../")
from utils.data_utils import (
    remove_reduntant_bracket,
    standardize_whitespace_v2,
    remove_redundant_array_env,
)


def process_args():
    parser = argparse.ArgumentParser(
        description="Preprocess (tokenize or normalize) latex formulas"
    )

    parser.add_argument(
        "--mode",
        "-m",
        dest="mode",
        choices=["tokenize", "normalize"],
        default="normalize",
        help=(
            "Tokenize (split to tokens seperated by space) or normalize (further translate to an equivalent standard form)."
        ),
    )
    parser.add_argument(
        "--input-file",
        "-i",
        dest="input_file",
        type=str,
        required=True,
        help=("Input file containing latex formulas. One formula per line."),
    )
    parser.add_argument(
        "--output-file",
        "-o",
        dest="output_file",
        type=str,
        required=True,
        help=("Output file."),
    )
    parser.add_argument(
        "--log-path",
        dest="log_path",
        type=str,
        default=None,
        help=("Log file path, default=log.txt"),
    )
    parameters = parser.parse_args()
    return parameters


def main():
    parameters = process_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s",
        filename=parameters.log_path,
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s"
    )
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    logging.info("Script being executed: %s" % __file__)

    input_file = parameters.input_file
    output_file = parameters.output_file

    assert os.path.exists(input_file), input_file
    shutil.copy(input_file, output_file)

    temp_file = output_file + ".tmp"

    with open(temp_file, "w") as fout:
        prepre = open(output_file, "r").read().replace("\r", " ")  # delete \r

        prepre = re.sub(r"\\raisebox\{[^\}]+\}", "", prepre, flags=re.S)
        prepre = re.sub(r"\\allowbreak", "", prepre, flags=re.S)

        # replace split, align with aligned
        prepre = re.sub(r"\\mathop", r"\\operatorname\*", prepre, flags=re.S)
        prepre = re.sub(r"\\noalign", "", prepre, flags=re.S)

        skip_ptn = "|".join(SKIP_TOK)
        prepre = re.sub(skip_ptn, "", prepre, flags=re.S)

        prepre = re.sub(r"\\textcolor\{[\w]+\}", "", prepre, flags=re.S)
        prepre = re.sub(r"\\textcolor\[[\w]+\]\{[\d,\.]+\}", "", prepre, flags=re.S)

        prepre = re.sub(
            r"\\begin{(split|align|alignedat|alignat|eqnarray|gather|gathered)\*?}(.+?)\\end{\1\*?}",
            r"\\begin{aligned}\2\\end{aligned}",
            prepre,
            flags=re.S,
        )
        prepre = re.sub(
            r"\\begin{d(cases|rcases)}(.+)\\end{\1}",
            r"\\begin{\1}\2\\end{\1}",
            prepre,
            flags=re.S,
        )
        prepre = re.sub(
            r"\\begin{(pmatrix|bmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix)\*}(\[[rlc]+\])(.+?)\\end{\1\*}",
            r"\\begin{\1}\3\\end{\1}",
            prepre,
            flags=re.S,
        )
        prepre = re.sub(
            r"\\begin{(smallmatrix)\*?}(.+?)\\end{\1\*?}",
            r"\\begin{matrix}\2\\end{matrix}",
            prepre,
            flags=re.S,
        )

        fout.write(prepre)

    cmd = r"cat %s | node %s %s > %s " % (
        temp_file,
        os.path.join(os.path.dirname(__file__), "standardize_latex.js"),
        parameters.mode,
        output_file,
    )
    ret = subprocess.call(cmd, shell=True)
    os.remove(temp_file)

    if ret != 0:
        logging.error("FAILED: %s" % cmd)

    temp_file = output_file + ".tmp"
    shutil.move(output_file, temp_file)

    total_expression = 0
    with open(input_file, "r") as fin:
        with open(output_file, "w") as fout:
            for line in fin:
                logging.info("====================================")
                logging.info(f"Before: {line}")
                if line.__contains__(
                    "\\genfrac"
                ):  # Do not have solution to normalize this token at the moment so ...
                    continue
                # #TODO: delete remaining line which have textcolor token that can not be resolved before
                if line.__contains__("\\textcolor"):
                    continue

                if line.__contains__("\\boxed"):
                    continue

                if line.__contains__("\\vcenter"):
                    continue

                if line.__contains__("\\tag"):
                    continue

                if line.__contains__("\\newcommand") or line.__contains__(
                    "\\renewcommand"
                ):
                    continue

                tokens = line.strip().split()
                # skip all sample contain overlapping and break logic
                continue_flag = False
                for tok in tokens:
                    if (
                        tok.__contains__("skip")
                        or tok.__contains__("break")
                        or tok.__contains__("smash")
                        or tok.__contains__("mathllap")
                        or tok.__contains__("mathrlap")
                        or tok.__contains__("mathclap")
                    ):
                        continue_flag = True
                        break
                if continue_flag:
                    continue

                tokens_out = []

                for token in tokens:
                    tokens_out.append(token)

                if len(tokens_out) > MIN_TOKENS:
                    post = remove_reduntant_bracket(tokens_out)
                    post = standardize_whitespace_v2(
                        post, STANDARD_WHITESPACE_SPACE, STANDARD_SPACE
                    )

                    # for op in OPERATORS:
                    #     if op in post:
                    #         operators = '\s?'.join(list(op))
                    #         operators = '\s?' + operators
                    #         if "lim" in op:
                    #             post = re.sub(r'\\%s'%op, r'\\operatorname\* {(%s)}' % operators, post)
                    #         else:
                    #             post = re.sub(r'\\%s'%op, r'\\operatorname {(%s)}' % operators, post)

                    # TODO: recheck if all font is converted to math font: textbf -> mathbf

                    for font in FONT:
                        post = post.replace(font, FONT[font])

                    for size_ in SIZE:
                        post = post.replace(size_, "")

                    post = post.replace(r"\\ \end{array}", r"\end{array}")
                    post = remove_redundant_array_env(post)

                    total_expression += 1
                    fout.write(post + "\n")
                    logging.info(f"After: {post}")
                else:
                    logging.info("Line too short")

    logging.info(f"Total expressions {total_expression}")
    os.remove(temp_file)


if __name__ == "__main__":
    main()
    logging.info("Jobs finished")
