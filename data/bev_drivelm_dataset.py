import os
import json
import random

import torch
from torch.utils.data import Dataset


class bev_drivelm_dataset(Dataset):
    def __init__(self, bev_folder_train, bev_folder_val, drivelm_json_paths):
        self.bev_folder_train = bev_folder_train
        self.bev_folder_val = bev_folder_val
        self.drivelm = []

        for drivelm_json_path in drivelm_json_paths:
            with open(drivelm_json_path, "r") as drivelm_json:
                self.drivelm_dict = json.load(drivelm_json)

            for scene_token, scene in self.drivelm_dict.items():
                for keyframe, keyframe_data in scene["key_frames"].items():
                    for q_type, qs in keyframe_data["QA"].items():
                        for q_dict in qs:
                            question = q_dict["Q"]
                            answer = q_dict["A"]
                            self.drivelm.append(
                                {
                                    "sample_token": keyframe,
                                    "question": question,
                                    "answer": answer,
                                    "scene_token": scene_token,
                                }
                            )

    def __len__(self):
        return len(self.drivelm)

    def __getitem__(self, index):
        drivelm_item = self.drivelm[index]
        sample_token = drivelm_item["sample_token"]

        # BEV -------------------------------------------
        bev_filename = os.path.join(self.bev_folder_train, sample_token + ".pt")
        det_filename = os.path.join(self.bev_folder_train, sample_token + "_det.pt")
        obj_filename = os.path.join(self.bev_folder_train, "obj/", sample_token + "_obj.pt")

        if not os.path.exists(bev_filename):
            bev_filename = os.path.join(self.bev_folder_val, sample_token + ".pt")
            det_filename = os.path.join(self.bev_folder_val, sample_token + "_det.pt")
            obj_filename = os.path.join(self.bev_folder_val, "obj/", sample_token + "_obj.pt")

        if not os.path.exists(bev_filename):
            return self.__getitem__(random.randint(0, len(self.drivelm) - 1))

        with open(bev_filename, "rb") as bev_file:
            bev = torch.load(bev_file)
            bev = bev.squeeze()

        with open(det_filename, "rb") as det_file:
            det = torch.load(det_file)
        
        with open(obj_filename, "rb") as obj_file:
            obj = torch.load(obj_file)

        return (
            bev,
            drivelm_item["question"],
            drivelm_item["answer"],
            drivelm_item["sample_token"],
            drivelm_item["scene_token"],
            det,
            obj
        )
