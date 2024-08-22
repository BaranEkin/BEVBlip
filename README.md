# Visual Question Answering for Traffic Environment Understanding
BEVBlip is an efficient and ligtweight Vision-Language Model (VLM) based on BLIP architecture, trained for comprehensive Visual Question Answering (VQA) task introduced by DriveLM on nuScenes dataset.
As the core idea, BEVBlip employs spatio-temporal Bird’s Eye View (BEV) maps acquired via BEVFormer as visual features and integrates visual and language features for enhanced traffic environment understanding.
In order to align BEV features with language, a pre-training stage utilizing GPT generated data is executed.

For an in depth explanation, please see: 

## Example Results
![image](https://github.com/user-attachments/assets/68bd5beb-916e-4383-8237-556f56c3d028)
![image](https://github.com/user-attachments/assets/034a191e-2597-4b4b-a310-7f27c8688333)

## Implementation
High-level outline of the proposed approach:

![image](https://github.com/user-attachments/assets/7b0c684a-2445-4e57-9804-08e4b30b5bf5)

### Pre-training
The architecture of the pre-training model: 

| ![image](https://github.com/user-attachments/assets/fb281f11-25d6-461d-a27f-31dd1b8e30db) |
|:--:| 
| *The bottom section illustrates offline data generation steps using BEVFormer and GPT-3.5. The upper right section shows the unified multimodal encoder-decoder with pretrained weights from BLIP. The upper left section depicts the compact vision transformer architecture, trained from scratch with BEV feature maps.* |

### Visual Question Answering
The architecture of the VQA model used for the fine-tuning on DriveLM task:

| ![image](https://github.com/user-attachments/assets/d9ceb4de-f8f4-4c54-81d5-c9484c60d582) |
|:--:| 
| *Left section shows the vision transformer, initialized with the weights from the pre-training stage. Right section illustrates the reconfiguration of text encoder and text decoder as question encoder and answer decoder respectively.* |

## Acknowledgement
Sources and references:
- BLIP: https://github.com/salesforce/BLIP
- BEVFormer: https://github.com/fundamentalvision/BEVFormer
- DriveLM: https://github.com/OpenDriveLab/DriveLM
