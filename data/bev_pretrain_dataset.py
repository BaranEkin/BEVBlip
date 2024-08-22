import os
import json
import glob

import torch
from torch.utils.data import Dataset


class bev_pretrain_dataset(Dataset):
    def __init__(
        self, bev_features_folder_path, scene_statements_path, nusc_sample_path
    ):
        with open(nusc_sample_path, "r") as nusc_sample_json:
            self.nusc_samples = json.load(nusc_sample_json)

        with open(scene_statements_path, "r") as scene_statements_json:
            self.scene_statements = json.load(scene_statements_json)

        self.bev_files_list = [
            file
            for file in glob.glob(os.path.join(bev_features_folder_path, "*.pt"))
            if not (file.endswith("_det.pt") or file.endswith("_obj.pt"))
        ]

    def get_scene_token(self, sample_token):
        for sample in self.nusc_samples:
            if sample["token"] == sample_token:
                return sample["scene_token"]

    def get_scene_statement(self, scene_token):
        for statement in self.scene_statements:
            if statement["scene_token"] == scene_token:
                return statement["statement"]

    def __len__(self):
        return len(self.bev_files_list)

    def __getitem__(self, index):
        # BEV -------------------------------------------
        bev_filename = self.bev_files_list[index]
        with open(bev_filename, "rb") as bev_file:
            bev = torch.load(bev_file)
            bev = bev.squeeze()

        # Statement -------------------------------------
        sample_idx = os.path.splitext(os.path.basename(bev_filename))[0]
        scene_token = self.get_scene_token(sample_idx)
        statement = self.get_scene_statement(scene_token)

        return bev, statement
