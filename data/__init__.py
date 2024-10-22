import torch
from torch.utils.data import DataLoader, random_split
from data.bev_pretrain_dataset import bev_pretrain_dataset
from data.bev_drivelm_dataset import bev_drivelm_dataset


def create_dataset(dataset, config):
    if dataset == "pretrain_bev":
        train_dataset = bev_pretrain_dataset(
            config["bev_features_folder_train"],
            config["scene_statements"],
            config["nuscenes_sample"],
        )
        val_dataset = bev_pretrain_dataset(
            config["bev_features_folder_val"],
            config["scene_statements"],
            config["nuscenes_sample"],
        )
        return train_dataset, val_dataset

    elif dataset == "bev_drivelm":
        bev_drivelm = bev_drivelm_dataset(
            config["bev_features_folder_train"],
            config["bev_features_folder_val"],
            config["drivelm_jsons"],
        )
        val_size = int(len(bev_drivelm) * 0.05)
        train_dataset, val_dataset = random_split(
            bev_drivelm,
            [len(bev_drivelm) - val_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        return train_dataset, val_dataset

    elif dataset == "bev_drivelm_split":
        train_dataset = bev_drivelm_dataset(
            config["bev_features_folder_train"],
            config["bev_features_folder_val"],
            config["train"],
        )

        val_dataset = bev_drivelm_dataset(
            config["bev_features_folder_train"],
            config["bev_features_folder_val"],
            config["val"],
        )

        return train_dataset, val_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            multiprocessing_context="spawn",
        )
        loaders.append(loader)
    return loaders
