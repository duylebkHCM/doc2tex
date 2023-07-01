import sys
import six
import lmdb
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import sys
from data.data_const import LMDB_CONST


class LMDB_Dataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.env = lmdb.open(
            root, max_readers=32, readonly=True, readahead=False, meminit=False
        )
        self.txn = self.env.begin(write=False)

        if not self.env:
            print("cannot create lmdb from %s" % (root))
            sys.exit(0)

    @property
    def dataset_samples(self):
        nSamples = int(self.txn.get(f"{LMDB_CONST.N_SAMPLES.value}".encode()))
        return nSamples

    @property
    def filtered_index_list(self):
        return [index + 1 for index in range(self.dataset_samples)]

    def __len__(self):
        return len(self.filtered_index_list)

    def __getitem__(self, index):
        assert index <= len(
            self
        ), f"index range error {index} with length of dataset {len(self)}"
        index = self.filtered_index_list[index]

        img_key = f"{LMDB_CONST.IMAGE.value}-%09d".encode() % index
        imgbuf = self.txn.get(img_key)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)

        label_key = f"{LMDB_CONST.LABEL.value}-%09d".encode() % index
        label = self.txn.get(label_key).decode("utf-8")
        name_key = f"{LMDB_CONST.PATH.value}-%09d".encode() % index
        img_name = self.txn.get(name_key).decode("utf-8")
        heightkey = f"{LMDB_CONST.HEIGHT.value}-%09d" % index
        widthkey = f"{LMDB_CONST.WIDTH.value}-%09d" % index
        height = self.txn.get(heightkey.encode())
        width = self.txn.get(widthkey.encode())
        h_img = np.frombuffer(height, dtype=np.int32)
        w_img = np.frombuffer(width, dtype=np.int32)

        img = Image.open(buf).convert("L")
        img = np.asarray(img).astype("uint8")

        return (img, label, img_name, h_img, w_img)
