import numpy as np
from torch.utils.data.sampler import Sampler
from functools import cached_property


class ClusterRandomSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    @cached_property
    def batch_lists(self):
        batch_lists = []
        all_cluster_indices = getattr(self.data_source, "cluster_batch_indices", [])

        if not len(all_cluster_indices):
            raise ValueError("Dataset do not contain any cluster")

        for _, cluster_indices in all_cluster_indices.items():
            if not len(cluster_indices):
                continue

            cluster_indices = np.array(cluster_indices, dtype=np.int32)
            if self.shuffle:
                p = np.random.permutation(len(cluster_indices))
            else:
                p = np.arange(len(cluster_indices))

            for i in range(0, len(cluster_indices), self.batch_size):
                batch = cluster_indices[p[i : i + self.batch_size]]
                if self.drop_last and (batch.shape[0] < self.batch_size):
                    continue
                batch_lists.append(batch)

        if not len(batch_lists):
            raise ValueError("Cannot sampling data sample from empty data source")

        final_batch_lists = np.empty(len(batch_lists), dtype=object)
        for idx, batch in enumerate(batch_lists):
            final_batch_lists[idx] = batch

        if self.shuffle:
            final_batch_lists = np.random.permutation(final_batch_lists)

        batch_indices = [batch.tolist() for batch in final_batch_lists]
        return batch_indices

    def __iter__(self):
        return iter(self.batch_lists)

    def __len__(self):
        return len(self.batch_lists)
