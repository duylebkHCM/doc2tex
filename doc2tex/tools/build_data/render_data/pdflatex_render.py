import argparse
import sys
import os
import glob
from tqdm.auto import tqdm
import cv2
from copy import deepcopy
import numpy as np
import pandas as pd
import logging
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from utils import crop_image
from pdflatex2png import Latex, tex2pil


def render_dataset(
    dataset: np.ndarray, unrendered: np.ndarray, dataset_type, args
) -> np.ndarray:
    """Renders a list of tex equations

    Args:
        dataset (numpy.ndarray): List of equations
        unrendered (numpy.ndarray): List of integers of size `dataset` that give the name of the saved image
        args (Union[Namespace, Munch]): additional arguments: mode (equation or inline), out (output directory), divable (common factor )
                                        batchsize (how many samples to render at once), dpi, font (Math font), preprocess (crop, alpha off)
                                        shuffle (bool)

    Returns:
        numpy.ndarray: equation indices that could not be rendered
    """
    assert len(unrendered) == len(
        dataset
    ), "unrendered and dataset must be of equal size"

    # remove successfully rendered equations
    if dataset_type == ".csv":
        rendered = np.array(
            [
                os.path.basename(img)
                for img in glob.glob(os.path.join(args.out, "*.png"))
            ]
        )
    else:
        rendered = np.array(
            [
                int(os.path.basename(img).split(".")[0])
                for img in glob.glob(os.path.join(args.out, "*.png"))
            ]
        )

    valid = [i for i, j in enumerate(unrendered) if j not in rendered]
    logging.info(f"Unrendered math {len(valid)}")

    # update unrendered and dataset
    dataset = dataset[valid]
    unrendered = unrendered[valid]

    order = (
        np.random.permutation(len(dataset)) if args.shuffle else np.arange(len(dataset))
    )
    faulty = []

    for batch_offset in tqdm(
        range(0, len(dataset), args.batchsize), desc="global batch index"
    ):
        batch = dataset[order[batch_offset : batch_offset + args.batchsize]]

        if len(batch) == 0:
            continue

        valid_math = np.asarray(
            [[i, x] for i, x in enumerate(batch) if x != ""], dtype=object
        )  # space used to prevent escape $

        dpi = args.dpi

        if len(valid_math) > 0:
            valid_idx, math = valid_math.T
            valid_idx = valid_idx.astype(np.int32)

            try:
                pngs, error_index = tex2pil(math, dpi=dpi, return_error_index=True)

                # error_index not count "" line, use valid_idx transfer to real index matching in batch index
                local_error_index = valid_idx[error_index]
                # tranfer in batch index to global batch index
                global_error_index = [batch_offset + _ for _ in local_error_index]

                faulty.extend(list(unrendered[order[global_error_index]]))

            except Exception as e:
                print("\n%s" % e, end="")

                faulty.extend(
                    list(
                        unrendered[order[batch_offset : batch_offset + args.batchsize]]
                    )
                )

                continue

            for inbatch_idx, order_idx in enumerate(
                range(batch_offset, batch_offset + args.batchsize)
            ):
                # exclude render failed equations and blank line
                if inbatch_idx in local_error_index or inbatch_idx not in valid_idx:
                    continue

                if ext == ".csv":
                    outpath = os.path.join(args.out, f"{unrendered[order[order_idx]]}")
                else:
                    outpath = os.path.join(
                        args.out,
                        f"{unrendered[order[order_idx]]}.png".zfill(MAX_LENGTH),
                    )

                logging.info(f"Save image {outpath}")

                png_idx = np.where(valid_idx == inbatch_idx)[0][0]

                if args.preprocess:
                    try:
                        data = np.asarray(pngs[png_idx])

                        # To invert the text to white
                        gray = 255 * (data[..., 0] < 128).astype(np.uint8)
                        white_pixels = np.sum(gray == 255)

                        # some png will be whole white, because some equation's syntax is wrong
                        # eg.$$ \mathit { \Iota \Kappa \Lambda \Mu \Nu \Xi \Omicron \Pi } $$
                        # extract from wikipedia english dump file https://dumps.wikimedia.org/enwiki/latest/

                        white_percentage = white_pixels / (
                            gray.shape[0] * gray.shape[1]
                        )
                        if white_percentage == 0:
                            continue

                        # Find all non-zero points (text)
                        coords = cv2.findNonZero(gray)

                        # Find minimum spanning bounding box
                        a, b, w, h = cv2.boundingRect(coords)
                        rect = data[b : b + h, a : a + w]
                        im = Image.fromarray(
                            (255 - rect[..., -1]).astype(np.uint8)
                        ).convert("L")
                        dims = []

                        for x in [w, h]:
                            div, mod = divmod(x, args.divable)
                            dims.append(args.divable * (div + (1 if mod > 0 else 0)))

                        offset_x, offset_y = (
                            np.random.randint(0, dims[0] - w) if dims[0] - w > 0 else 0,
                            np.random.randint(0, dims[1] - h) if dims[1] - h > 0 else 0,
                        )
                        padded = Image.new("L", dims, 255)
                        padded.paste(
                            im,
                            (
                                offset_x,
                                offset_y,
                                offset_x + im.size[0],
                                offset_y + im.size[1],
                            ),
                        )
                        padded.save(outpath)

                    except Exception as e:
                        print(e)
                        pass
                else:
                    np_img = np.asarray(pngs[png_idx])
                    img = Image.fromarray(np_img).convert("L")
                    success = crop_image(img, outpath)
                    if success:
                        img.save(outpath)

    # prevent repeat between two error_index and imagemagic error
    faulty = list(set(faulty))
    faulty.sort()

    return np.array(faulty)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render dataset")
    parser.add_argument(
        "-i", "--data", type=str, required=True, help="file of list of latex code"
    )
    parser.add_argument("-o", "--out", type=str, required=True, help="output directory")
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=100,
        help="How many equations to render at once",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["inline", "equation"],
        default="equation",
        help="render as inline or equation",
    )
    parser.add_argument("--dpi", type=int, default=200, help="dpi range to render in")
    parser.add_argument(
        "-p",
        "--no-preprocess",
        dest="preprocess",
        default=True,
        action="store_false",
        help="crop, remove alpha channel, padding",
    )
    parser.add_argument(
        "-d", "--divable", type=int, default=32, help="To what factor to pad the images"
    )
    parser.add_argument(
        "-s",
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the equations in the first iteration",
    )
    parser.add_argument("-l", "--log-name", default="render_log.txt", help="")
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.out, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s",
        filename=str(Path(args.out).parent.joinpath(args.log_name)),
    )
    logging.getLogger("RENDER-LOGGER")

    data_path = Path(args.data)
    ext = data_path.suffix
    if ext == ".csv":
        df = pd.read_csv(args.data, sep="\t")
        dataset = df["label"].values
        unrendered = df["id"].values
        full_names = deepcopy(unrendered)
    else:
        dataset = np.array(open(args.data, "r").read().split("\n"), dtype=object)
        dataset = dataset[:-1]
        unrendered = np.arange(len(dataset))
    # sys.exit(0)
    MAX_LENGTH = len(str(len(dataset)))
    print("TOTAL MATH", len(dataset))

    failed = np.array([])

    round = 0
    while unrendered.tolist() != failed.tolist():
        logging.info("===========================")
        failed = unrendered

        if not len(unrendered):
            break

        if ext == ".csv":
            if len(full_names) == len(unrendered):
                unrender_idx = np.arange(len(full_names))
            else:
                unrender_idx = []
                for i, j in enumerate(full_names.tolist()):
                    if j in unrendered.tolist():
                        unrender_idx.append(i)
                unrender_idx = np.array(unrender_idx)
        else:
            unrender_idx = unrendered

        unrendered = render_dataset(dataset[unrender_idx], unrendered, ext, args)

        logging.info(f"Number remain of round {round}: {len(unrendered)}")

        for img in unrendered:
            logging.info(f"Unrendered img {img}")

        if len(unrendered) < 50 * args.batchsize:
            args.batchsize = max([1, args.batchsize // 2])

        args.shuffle = True
