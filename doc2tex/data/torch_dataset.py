import numpy as np
from collections import defaultdict
from functools import cached_property
from typing import List, Dict, Any

from .helpers import get_size
from .data_const import LMDB_CONST
from .lmdb_dataset import LMDB_Dataset


class Im2LaTeXDataset(LMDB_Dataset):
    def __init__(self, root, config):
        super(Im2LaTeXDataset, self).__init__(root, config)

    @cached_property
    def filtered_index_list(self):
        if self.config["data_filtering_off"]:
            filtered_index_list = [index + 1 for index in range(self.dataset_samples)]
        else:
            filtered_index_list = []
            for index in range(self.dataset_samples):
                index += 1  # lmdb starts with 1
                label_key = "label-%09d".encode() % index
                label = self.txn.get(label_key).decode("utf-8")

                if len(label) > self.config["batch_max_length"]:
                    continue

                filtered_index_list.append(index)

        return filtered_index_list

    def create_bucket(self, idx):
        heightkey = f"{LMDB_CONST.HEIGHT.value}-%09d" % idx
        widthkey = f"{LMDB_CONST.WIDTH.value}-%09d" % idx
        height = self.txn.get(heightkey.encode())
        width = self.txn.get(widthkey.encode())
        h_img = np.fromstring(height, dtype=np.int32)
        w_img = np.fromstring(width, dtype=np.int32)
        h_img = int(h_img[0])
        w_img = int(w_img[0])
        assert isinstance(h_img, int) and isinstance(w_img, int)
        h_img, w_img = get_size(h_img, w_img, self.config)
        return h_img, w_img

    @cached_property
    def cluster_batch_indices(self) -> List[Dict[Any, List[int]]]:
        cluster_indices = defaultdict(list)
        exclude = 0

        for i, idx in enumerate(self.filtered_index_list):
            h, w = self.create_bucket(idx)
            min_h, min_w = self.config["min_dimension"]
            max_h, max_w = self.config["max_dimension"]

            if min_h <= h <= max_h and min_w <= w <= max_w:
                cluster_indices[(h, w)].append(i)
            else:
                exclude += 1
                continue

        assert sum(len(val) for k, val in cluster_indices.items()) == (
            len(self.filtered_index_list) - exclude
        )

        return cluster_indices

    def _get_new_size(self, index):
        for k in self.cluster_batch_indices:
            if index in self.cluster_batch_indices[k]:
                return k
