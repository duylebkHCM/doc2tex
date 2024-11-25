import sys
import six
import lmdb
from PIL import Image
import numpy as np
from functools import cached_property
from torch.utils.data import Dataset
import sys
from .data_const import LMDB_CONST


class LMDB_Dataset(Dataset):
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.env = lmdb.open(
            root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(write=False)

        if not self.env:
            print("cannot create lmdb from %s" % (root))
            sys.exit(0)

    @cached_property
    def dataset_samples(self):
        nSamples = int(self.txn.get(f"{LMDB_CONST.N_SAMPLES.value}".encode()))
        return nSamples

    @cached_property
    def filtered_index_list(self):
        return [index + 1 for index in range(self.dataset_samples)]

    def _get_new_size(self, index):
        return None, None

    def __len__(self):
        return len(self.filtered_index_list)

    def __getitem__(self, index):
        assert index <= len(
            self
        ), f"index range error {index} with length of dataset {len(self)}"
        value = self.filtered_index_list[index]

        label_key = f"{LMDB_CONST.LABEL.value}-%09d".encode() % value
        label = self.txn.get(label_key).decode("utf-8")
        img_key = f"{LMDB_CONST.IMAGE.value}-%09d".encode() % value
        imgbuf = self.txn.get(img_key)
        name_key = f"{LMDB_CONST.PATH.value}-%09d".encode() % value
        img_name = self.txn.get(name_key).decode("utf-8")

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)

        try:
            if self.config["rgb"]:
                img = Image.open(buf).convert("RGB")  # for color image
            else:
                img = Image.open(buf).convert("L")
        except IOError:
            print(f"Corrupted image for {value}")
            # make dummy image and dummy label for corrupted image.
            if self.config["rgb"]:
                img = Image.new("RGB", (self.config["imgW"], self.config["imgH"]))
            else:
                img = Image.new("L", (self.config["imgW"], self.config["imgH"]))
            label = "[dummy_label]"

        if self.config.get("downsample", None) is not None:
            ori_h, ori_w = img.size[::-1]
            ratio = self.config["downsample"]
            if (
                ori_h / ratio >= self.config["min_dimension"][0]
                and ori_w / ratio >= self.config["min_dimension"][1]
            ):
                ori_h, ori_w = (
                    ori_h / self.config["downsample"],
                    ori_w / self.config["downsample"],
                )
                img = img.resize((ori_w, ori_h), resample=Image.LANCZOS)

        img = np.asarray(img).astype("uint8")

        new_h, new_w = self._get_new_size(index)

        return (img, label, (new_h, new_w), img_name)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + ": ("
            + f"Number of samples: {len(self)}, Data path: {self.root}"
            + ")"
        )
