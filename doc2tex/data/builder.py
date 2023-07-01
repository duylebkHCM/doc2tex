import torch
from data.torch_dataset import Im2LaTeXDataset as Torch_Im2LaTeXDataset
from data.sampler import ClusterRandomSampler
from data.collate_fn import ClusterCollate
from data.prefetcher import PrefetchLoader
from transform.math_transform import Math_Transform


def build_loader(config: dict, device: torch.device):
    train_dataset = Torch_Im2LaTeXDataset(config["train_data"], config)
    collate_train = ClusterCollate(config, image_padding_value=255)
    sampler = ClusterRandomSampler(
        train_dataset, config["batch_size"], True, not config["keep_smaller_batches"]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=int(config["workers"]),
        collate_fn=collate_train,
        pin_memory=True,
    )

    train_loader = PrefetchLoader(train_loader, device=device)

    collate_val = ClusterCollate(
        opt=config, transform_img=None, image_padding_value=255
    )
    valid_dataset = Torch_Im2LaTeXDataset(root=config["valid_data"], config=config)
    sampler = ClusterRandomSampler(
        valid_dataset, config["batch_size"], True, not config["keep_smaller_batches"]
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_sampler=sampler,
        num_workers=int(config["workers"]),
        collate_fn=collate_val,
        pin_memory=True,
    )

    valid_loader = PrefetchLoader(valid_loader, device=device)

    return train_loader, valid_loader, Math_Transform(config)
