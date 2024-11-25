import sys, logging, argparse, os, pandas as pd, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.data_const import LABEL_KEY
from sklearn.model_selection import train_test_split


def process_args(args):
    parser = argparse.ArgumentParser(description="Generate train, val, test file.")

    parser.add_argument(
        "--img-path",
        dest="img_paths",
        type=str,
        required=True,
        help=("Directory of rendered images"),
    )
    parser.add_argument(
        "--tex-path",
        dest="tex_path",
        type=str,
        required=True,
        help=("Path of normalized latex file used to render images"),
    )
    parser.add_argument(
        "--output-label",
        dest="output_label",
        type=str,
        required=True,
        help="Output path of all label files",
    )
    parser.add_argument(
        "--log-path",
        dest="log_path",
        type=str,
        default="log.txt",
        help=("Log file path, default=log.txt"),
    )
    parser.add_argument(
        "--split-ratio",
        dest="ratio",
        type=float,
        nargs="+",
        default=None,
        help="Dataset split ratio between train, valid, test",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        default=False,
        help="Whether split dataset, used for debugging stage.",
    )
    parameters = parser.parse_args(args)
    return parameters


def main(args):
    parameters = process_args(args)
    label_path = os.path.join(parameters.output_label, LABEL_KEY.DATETIME_FMT.value)
    if not os.path.exists(label_path):
        os.makedirs(label_path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s",
        filename=os.path.join(label_path, parameters.log_path),
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s"
    )
    console.setFormatter(formatter)
    logging.getLogger("LABEL-LOGGER").addHandler(console)

    logging.info("Script being executed: %s" % __file__)

    img_path = parameters.img_path
    assert os.path.exists(img_path), img_path
    tex_path = parameters.tex_path
    assert os.path.exists(tex_path), tex_path

    saved_info = {LABEL_KEY.IMAGE_ID.value: [], LABEL_KEY.LABEL.value: []}
    start_load_time = time.time()
    formulas = [line.strip() for line in open(tex_path).readlines()]
    print(
        len(formulas)
    )  # default read in UTF-8 mode for Unix, check with import locale; locale.getpreferredencoding(False)
    logging.info(f"Loadding formulas time {time.time()-start_load_time} s")

    try:
        for img_name in os.listdir(img_path):
            order = int(os.path.splitext(img_name)[0])
            lbl = formulas[order]
            saved_info[LABEL_KEY.IMAGE_ID.value].append(img_name)
            saved_info[LABEL_KEY.LABEL.value].append(lbl)
    except Exception:
        print("order", order)

    total_df = pd.DataFrame(data=saved_info)
    if parameters.split:
        if parameters.ratio:
            train_rt, val_rt, test_rt = tuple(parameters.ratio)
        else:
            train_rt, val_rt, test_rt = 0.8, 0.1, 0.1

        train_df, test_df = train_test_split(
            total_df, test_size=test_rt, random_state=42
        )
        train_df, val_df = train_test_split(
            train_df, test_size=val_rt / (train_rt + val_rt), random_state=42
        )

        def process_df(df, part):
            df.sample(frac=1, random_state=42).reset_index(drop=True, inplace=True)
            df.to_csv(os.path.join(label_path, f"{part}.csv"), index=False, header=True)
            logging.info(f"{part} has {len(df)} samples")

        process_df(train_df, "train")
        process_df(val_df, "valid")
        process_df(test_df, "test")
    else:
        total_df.to_csv(os.path.join(label_path, "total.csv"), index=False, header=True)


if __name__ == "__main__":
    main(sys.argv[1:])
    logging.info("Jobs finished")
