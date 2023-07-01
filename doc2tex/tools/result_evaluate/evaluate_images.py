#!/usr/bin/env python3
from pathlib import Path
import sys, os, argparse, logging, glob
import numpy as np
from PIL import Image
import distance
import itertools
from LevSeq import StringMatcher
import pandas as pd


def process_args(args):
    parser = argparse.ArgumentParser(description="Evaluate image related metrics.")
    parser.add_argument(
        "--export_csv",
        action="store_true",
        default=False,
        help="export metric of each imgae in csv file",
    )
    parser.add_argument(
        "--csv_dir", default="", help="csv path of result metric in case export csv"
    )
    parser.add_argument(
        "--images-gold",
        dest="images_gold",
        type=str,
        required=True,
        help="Images directory containing ground truth images",
    )
    parser.add_argument(
        "--images-pred",
        dest="images_pred",
        type=str,
        required=True,
        help=(
            'Images directory containing the rendered images. A subfolder with name "images_gold" for the rendered gold images, and a subfolder "images_pred" must be created beforehand by using scripts/evaluation/render_latex.py.'
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--log-path",
        dest="log_path",
        type=str,
        default="log_evaluate.txt",
        help=("Log file path, default=log.txt"),
    )
    parameters = parser.parse_args(args)
    return parameters


def main(args):
    parameters = process_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s",
        filename=os.path.join(parameters.out_dir, parameters.log_path),
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s"
    )
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    logging.info("Script being executed: %s" % __file__)
    uf = open(os.path.join(parameters.out_dir, "unmatched_filenames.txt"), "w")

    images_pred = parameters.images_pred
    images_gold = parameters.images_gold
    assert os.path.exists(images_gold), images_gold
    assert os.path.exists(images_pred), images_pred

    total_edit_distance = 0
    total_ref = 0
    total_num = 0
    total_correct = 0
    total_correct_eliminate = 0

    ref_filenames = glob.glob(os.path.join(images_gold, "*.png"))

    unmatched_filenames = []

    if parameters.export_csv and parameters.csv_dir is not None:
        result_df = pd.read_csv(
            parameters.csv_dir,
            names=["name", "pred", "label", "ed", "word_ed", "bleu", "iscorrect"],
        )

    for filename in ref_filenames:
        base_filename = os.path.basename(filename)
        pred_filename = os.path.join(images_pred, filename.split(os.sep)[-1])
        edit_distance, ref, match1, match2 = img_edit_distance_file(
            filename, pred_filename
        )
        total_edit_distance += edit_distance
        result_df.loc[result_df["name"] == base_filename, "img_distance"] = (
            edit_distance / ref
        )
        total_ref += ref
        total_num += 1

        if match1:
            result_df.loc[result_df["name"] == base_filename, "match_w_space"] = 1
            total_correct += 1
        else:
            result_df.loc[result_df["name"] == base_filename, "match_w_space"] = 0

        if match2:
            result_df.loc[result_df["name"] == base_filename, "match_wo_space"] = 1
            total_correct_eliminate += 1
        else:
            result_df.loc[result_df["name"] == base_filename, "match_wo_space"] = 0

        if not (match1 or match2):
            unmatched_filenames.append(filename)

        if total_num % 100 == 0:
            logging.info("Total Num: %d" % total_num)
            logging.info("Accuracy (w spaces): %f" % (float(total_correct) / total_num))
            logging.info(
                "Accuracy (w/o spaces): %f"
                % (float(total_correct_eliminate) / total_num)
            )
            logging.info(
                "Edit Dist (w spaces): %f"
                % (1.0 - float(total_edit_distance) / total_ref)
            )
            logging.info("Total Correct (w spaces): %d" % total_correct)
            logging.info("Total Correct (w/o spaces): %d" % total_correct_eliminate)
            logging.info("Total Edit Dist (w spaces): %d" % total_edit_distance)
            logging.info("Total Ref (w spaces): %d" % total_ref)
            logging.info("")

            for file in unmatched_filenames:
                uf.write("%s\n" % file)
            unmatched_filenames = []
            uf.flush()
            os.fsync(uf.fileno())

    new_name = Path(parameters.csv_dir).stem.split(".")[0] + "_img_metric.csv"
    result_df.to_csv(
        str(Path(parameters.csv_dir).parent / new_name), index=False, header=True
    )
    logging.info("------------------------------------")
    logging.info("Final")
    logging.info("Total Num: %d" % total_num)
    logging.info("Accuracy (w spaces): %f" % (float(total_correct) / total_num))
    logging.info(
        "Accuracy (w/o spaces): %f" % (float(total_correct_eliminate) / total_num)
    )
    logging.info(
        "Edit Dist (w spaces): %f" % (1.0 - float(total_edit_distance) / total_ref)
    )
    logging.info("Total Correct (w spaces): %d" % total_correct)
    logging.info("Total Correct (w/o spaces): %d" % total_correct_eliminate)
    logging.info("Total Edit Dist (w spaces): %d" % total_edit_distance)
    logging.info("Total Ref (w spaces): %d" % total_ref)

    for file in unmatched_filenames:
        uf.write("%s\n" % file)
    uf.flush()
    os.fsync(uf.fileno())


def trim_image(np_ar):
    """
    Trims empty rows and columns of an image - i.e. those that have all pixels == 255
    :param np_ar: numpy.ndarray of a image of shape (H,W). Each value should have value between 0 and 255.
    :returns: numpy array of the trimmed image, shape (H', W') where H' <= H and W' <= W
    """
    rows = [(row == 255).all() for row in np_ar]
    cols = [(row == 255).all() for row in np_ar.transpose()]

    top = len([x for x in itertools.takewhile(lambda x: x, rows)])
    bottom = len(rows) - len([x for x in itertools.takewhile(lambda x: x, rows[::-1])])

    left = len([x for x in itertools.takewhile(lambda x: x, cols)])
    right = len(cols) - len([x for x in itertools.takewhile(lambda x: x, cols[::-1])])

    if top != 0 or left != 0 or bottom != len(rows) or right != len(cols):
        print(
            "Trimmed image from shape (%d, %d) to (%d, %d)"
            % (len(rows), len(cols), bottom - top, right - left)
        )

    return np_ar[top:bottom, left:right]


# return (edit_distance, ref, match, match w/o)
def img_edit_distance(im1, im2, out_path=None):
    img_data1 = np.asarray(im1, dtype=np.uint8)  # height, width
    img_data1 = trim_image(np.transpose(img_data1))

    h1 = img_data1.shape[1]
    w1 = img_data1.shape[0]

    img_data1 = (img_data1 <= 128).astype(np.uint8)

    if im2:
        img_data2 = np.asarray(im2, dtype=np.uint8)  # height, width
        img_data2 = trim_image(np.transpose(img_data2))
        h2 = img_data2.shape[1]
        w2 = img_data2.shape[0]
        img_data2 = (img_data2 <= 128).astype(np.uint8)

    else:
        img_data2 = []
        h2 = h1

    if h1 == h2:
        seq1 = ["".join([str(i) for i in item]) for item in img_data1]
        seq2 = ["".join([str(i) for i in item]) for item in img_data2]

    elif h1 > h2:  # pad h2
        seq1 = ["".join([str(i) for i in item]) for item in img_data1]
        seq2 = [
            "".join([str(i) for i in item]) + "".join(["0"] * (h1 - h2))
            for item in img_data2
        ]

    else:
        seq1 = [
            "".join([str(i) for i in item]) + "".join(["0"] * (h2 - h1))
            for item in img_data1
        ]
        seq2 = ["".join([str(i) for i in item]) for item in img_data2]

    seq1_int = [int(item, 2) for item in seq1]
    seq2_int = [int(item, 2) for item in seq2]

    big = int("".join(["0" for i in range(max(h1, h2))]), 2)
    seq1_eliminate = []
    seq2_eliminate = []
    seq1_new = []
    seq2_new = []

    for idx, items in enumerate(seq1_int):
        if items > big:
            seq1_eliminate.append(items)
            seq1_new.append(seq1[idx])
    for idx, items in enumerate(seq2_int):
        if items > big:
            seq2_eliminate.append(items)
            seq2_new.append(seq2[idx])

    if len(seq2) == 0:
        return (len(seq1), len(seq1), False, False)

    def make_strs(int_ls, int_ls2):
        d = {}
        seen = []

        def build(ls):
            for l in ls:
                if int(l, 2) in d:
                    continue
                found = False
                l_arr = np.array(list(map(int, l)))

                for l2, l2_arr in seen:
                    if np.abs(l_arr - l2_arr).sum() < 5:
                        d[int(l, 2)] = d[int(l2, 2)]
                        found = True
                        break
                if not found:
                    d[int(l, 2)] = chr(len(seen))
                    seen.append((l, np.array(list(map(int, l)))))

        build(int_ls)
        build(int_ls2)

        return "".join([d[int(l, 2)] for l in int_ls]), "".join(
            [d[int(l, 2)] for l in int_ls2]
        )

    # if out_path:
    seq1_t, seq2_t = make_strs(seq1, seq2)

    edit_distance = distance.levenshtein(seq1_int, seq2_int)

    match = True

    if edit_distance > 0:
        matcher = StringMatcher(None, seq1_t, seq2_t)

        ls = []
        for op in matcher.get_opcodes():
            if op[0] == "equal" or (op[2] - op[1] < 5):
                ls += [[int(r) for r in l] for l in seq1[op[1] : op[2]]]

            elif op[0] == "replace":
                a = seq1[op[1] : op[2]]
                b = seq2[op[3] : op[4]]
                ls += [
                    [
                        int(r1) * 3 + int(r2) * 2 if int(r1) != int(r2) else int(r1)
                        for r1, r2 in zip(
                            a[i] if i < len(a) else [0] * 1000,
                            b[i] if i < len(b) else [0] * 1000,
                        )
                    ]
                    for i in range(max(len(a), len(b)))
                ]
                match = False

            elif op[0] == "insert":
                ls += [[int(r) * 3 for r in l] for l in seq2[op[3] : op[4]]]
                match = False

            elif op[0] == "delete":
                match = False
                ls += [[int(r) * 2 for r in l] for l in seq1[op[1] : op[2]]]

    match1 = match

    seq1_t, seq2_t = make_strs(seq1_new, seq2_new)

    if len(seq2_new) == 0 or len(seq1_new) == 0:
        if len(seq2_new) == len(seq1_new):
            return (
                edit_distance,
                max(len(seq1_int), len(seq2_int)),
                match1,
                True,
            )  # all blank
        return (edit_distance, max(len(seq1_int), len(seq2_int)), match1, False)

    match = True
    matcher = StringMatcher(None, seq1_t, seq2_t)

    ls = []
    for op in matcher.get_opcodes():
        if op[0] == "equal" or (op[2] - op[1] < 5):
            ls += [[int(r) for r in l] for l in seq1[op[1] : op[2]]]
        elif op[0] == "replace":
            a = seq1[op[1] : op[2]]
            b = seq2[op[3] : op[4]]
            ls += [
                [
                    int(r1) * 3 + int(r2) * 2 if int(r1) != int(r2) else int(r1)
                    for r1, r2 in zip(
                        a[i] if i < len(a) else [0] * 1000,
                        b[i] if i < len(b) else [0] * 1000,
                    )
                ]
                for i in range(max(len(a), len(b)))
            ]
            match = False
        elif op[0] == "insert":
            ls += [[int(r) * 3 for r in l] for l in seq2[op[3] : op[4]]]
            match = False
        elif op[0] == "delete":
            match = False
            ls += [[int(r) * 2 for r in l] for l in seq1[op[1] : op[2]]]

    match2 = match

    return (edit_distance, max(len(seq1_int), len(seq2_int)), match1, match2)


def img_edit_distance_file(file1, file2, output_path=None):
    img1 = Image.open(file1).convert("L")
    if os.path.exists(file2):
        img2 = Image.open(file2).convert("L")
    else:
        img2 = None
    return img_edit_distance(img1, img2, output_path)


if __name__ == "__main__":
    main(sys.argv[1:])
    logging.info("Jobs finished")
