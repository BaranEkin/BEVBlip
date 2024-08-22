import torch
from torch.utils.data.dataloader import DataLoader

from data.bev_drivelm_dataset import bev_drivelm_dataset
from models.blip_bev_vqa import BLIP_BEV_VQA
from eval.drivelm.utils import generate_drivelm_output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = bev_drivelm_dataset(
    "./data_thesis/bev_features_t/train",
    "./data_thesis/bev_features_t/val",
    "./data_thesis/QA_dataset_nus/test_converted.json",
)


data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

print("Initializing the model...")
model = BLIP_BEV_VQA(use_vit=True, use_det=True, use_obj=True)
model = model.to(device)
print("Model initialized.")

print("Loading previous checkpoint...")
checkpoint = torch.load(
    "/workspace/thesis/output/BEV_VQA_DriveLM/BLIP_BEV_VQA_DriveLM_v6_obj_feats_23.pth"
)
model.load_state_dict(checkpoint["model"])
print("Previous checkpoint loaded.")

print("Starting inference...")
generate_drivelm_output(model, data_loader, "v6_test_converted", device)
print("Inference complete!")
