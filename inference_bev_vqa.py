import torch
from data.bev_drivelm_dataset import bev_drivelm_dataset
from eval.drivelm.utils import generate_drivelm_output
from models.blip_bev_vqa import BLIP_BEV_VQA
from torch.utils.data.dataloader import DataLoader


device = torch.device("cuda")

dataset = bev_drivelm_dataset(
    "/workspace/BLIP/data_thesis/bev_features_t/train",
    "/workspace/BLIP/data_thesis/bev_features_t/val",
    ["/workspace/BLIP/data_thesis/QA_dataset_nus/v1_1_val_nus_q_only.json"],
)


data_loader = DataLoader(dataset, batch_size=1, shuffle=False)


print("Initializing the model...")
model = BLIP_BEV_VQA()
model = model.to(device)
print("Model initialized.")

print("Loading previous checkpoint...")
checkpoint = torch.load(
    "/workspace/BLIP/output/BEV_VQA_DriveLM/BLIP_BEV_VQA_DriveLM_v4_new_bev_26.pth"
)
model.load_state_dict(checkpoint["model"])
print("Previous checkpoint loaded!")

print("Starting inference...")
generate_drivelm_output(model, data_loader, "v4", device)
print("Inference complete!")
