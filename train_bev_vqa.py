import argparse
import os
import yaml
import numpy as np
import random
import time
from datetime import datetime, timedelta
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split


from models.blip_bev_vqa import BLIP_BEV_VQA
from models.blip_bev_pretrain import BLIP_BEV_Pretrain
import utils
from utils import warmup_lr_schedule, step_lr_schedule
from data import create_dataset, create_sampler, create_loader
from eval_blip_bev import LanguageEvaluation


def train(
    model, data_loader, optimizer, epoch, device, config, writer, gen_log, gen_freq
):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 50
    num_samples = len(data_loader)

    data_loader.sampler.set_epoch(epoch)

    for i, (bev, question, answer, _, _, det, obj) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        optimizer.zero_grad()

        bev = bev.to(device, non_blocking=True)
        loss = model(question=question, answer=answer, bev=bev, det=det, obj=obj)

        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        global_step = epoch * len(data_loader) + i

        writer.add_scalar("Train/Loss", loss.item(), global_step)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], global_step)

        if i % gen_freq == 0:
            generate_log_entry(
                model, bev, question, det, obj, answer, epoch, i, gen_log, mode="Train"
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {
        k: "{:.3f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }


def validation(model, data_loader, epoch, device, writer, gen_log, gen_freq):
    print(f"\n[EPOCH: {epoch}] Starting validation...\n")

    model.eval()
    with torch.no_grad():
        lang_metrics = {
            "Bleu_1": 0.0,
            "Bleu_2": 0.0,
            "Bleu_3": 0.0,
            "Bleu_4": 0.0,
            "ROUGE_L": 0.0,
            "CIDEr": 0.0,
        }

        num_lang_fail = 0
        num_samples = len(data_loader)

        for i, (bev, question, answer, _, _, det, obj) in enumerate(data_loader):
            print(f"\rRunning Validation {i + 1}/{num_samples}", end="")
            bev = bev.to(device, non_blocking=True)
            output = model.generate(question, bev=bev, det=det, obj=obj)

            # LANGUAGE METRICS CALCULATION
            try:
                lang_scores = LanguageEvaluation.evaluate(list(output), list(answer))
                lang_metrics["Bleu_1"] += float(lang_scores["Bleu_1"])
                lang_metrics["Bleu_2"] += float(lang_scores["Bleu_2"])
                lang_metrics["Bleu_3"] += float(lang_scores["Bleu_3"])
                lang_metrics["Bleu_4"] += float(lang_scores["Bleu_4"])
                lang_metrics["ROUGE_L"] += float(lang_scores["ROUGE_L"])
                lang_metrics["CIDEr"] += float(lang_scores["CIDEr"])
            except:
                num_lang_fail += 1

            if i % gen_freq == 0:
                generate_log_entry(
                    model, bev, question, det, obj, answer, epoch, i, gen_log, mode="Val"
                )

        # Averaging over epoch
        for m in lang_metrics:
            lang_metrics[m] = lang_metrics[m] / max(1, num_samples - num_lang_fail)

        writer.add_scalar("Val/BLEU-1", lang_metrics["Bleu_1"], epoch)
        writer.add_scalar("Val/BLEU-2", lang_metrics["Bleu_2"], epoch)
        writer.add_scalar("Val/BLEU-3", lang_metrics["Bleu_3"], epoch)
        writer.add_scalar("Val/BLEU-4", lang_metrics["Bleu_4"], epoch)
        writer.add_scalar("Val/ROUGE-L", lang_metrics["ROUGE_L"], epoch)
        writer.add_scalar("Val/CIDEr", lang_metrics["CIDEr"], epoch)
        print("\nLanguage metric fails during validation:", num_lang_fail)
        print(f"\n[EPOCH: {epoch}] Validation complete!\n")
        model.train()


def generate_log_entry(model, bev, question, det, obj, answer, ep, step, log_file, mode="Train"):
    output = model.generate(question, bev=bev, det=det, obj=obj)
    print(f"\nEpoch: {ep}, Mode: {mode}, Step: {step}", file=log_file, flush=True)
    print(f"Q:  {question[0]}", file=log_file, flush=True)
    print(f"GT: {answer[0]}", file=log_file, flush=True)
    print(f"Out:{output[0]}", file=log_file, flush=True)


def main(args, config):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating datasets")
    train_dataset, val_dataset = create_dataset("bev_drivelm_splitted", config)

    print("number of training samples: %d" % len(train_dataset))
    print("number of validation samples: %d" % len(val_dataset))
    
    """val_dataset = torch.utils.data.Subset(val_dataset, range(3000))
    print("number of selected validation samples: %d" % len(val_dataset))"""

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    train_sampler = create_sampler([train_dataset], [True], num_tasks, global_rank)
    val_sampler = create_sampler([val_dataset], [True], num_tasks, global_rank)

    train_loader = create_loader(
        [train_dataset],
        train_sampler,
        batch_size=[config["batch_size"]],
        num_workers=[4],
        is_trains=[True],
        collate_fns=[None],
    )[0]
    val_loader = create_loader(
        [val_dataset],
        val_sampler,
        batch_size=[config["batch_size"]],
        num_workers=[4],
        is_trains=[True],
        collate_fns=[None],
    )[0]

    #### Model ####
    print("Initializing the model...")
    model = BLIP_BEV_VQA(use_vit=True, use_det=True, use_obj=True)
    model = model.to(device)
    print("Model initialized.")

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config["init_lr"],
        weight_decay=config["weight_decay"],
    )

    run_name = "BLIP_BEV_VQA_DriveLM_notpretrained_skip_obj"
    todays_date = datetime.now().strftime("%d-%m")
    sum_writer = SummaryWriter(log_dir=f"runs/{todays_date}_{run_name}")

    start_new = True
    if start_new:
        start_epoch = 1
        # START FROM PRETRAINED WEIGHTS --------------------
        print("Loading pretrained weights...")
        checkpoint = torch.load(
            r"/workspace/thesis/output/BEV_VQA_DriveLM/BLIP_BEV_VQA_DriveLM_notpretrained_skip_obj_raw_3.pth"
        )
        model.load_state_dict(checkpoint["model"], strict=False)
        print("Pretrained weights loaded!")
        # ----------------------------------------------------

    else:
        # CONTINUE FROM CHECKPOINT ----------------------------
        print("Loading previous checkpoint...")
        checkpoint = torch.load(
            r"/workspace/BLIP/output/BEV_VQA_DriveLM/BLIP_BEV_VQA_DriveLM_bs10_lr5e-6_4.pth"
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print("Previous checkpoint loaded!")
        # ----------------------------------------------------

    with open(f"./logs/log_{todays_date}_{run_name}.txt", "w") as gen_log_file:
        validation(model, val_loader, 0, device, sum_writer, gen_log_file, gen_freq=2)

        print("Start training")
        start_time = time.time()

        for epoch in range(start_epoch, config["max_epoch"]):
            step_lr_schedule(
                optimizer,
                epoch,
                config["init_lr"],
                config["min_lr"],
                config["lr_decay_rate"],
            )

            train_stats = train(
                model,
                train_loader,
                optimizer,
                epoch,
                device,
                config,
                sum_writer,
                gen_log_file,
                gen_freq=100,
            )
            validation(
                model, val_loader, epoch, device, sum_writer, gen_log_file, gen_freq=2
            )

            if utils.is_main_process():
                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    "epoch": epoch,
                }
                save_obj = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                }
                torch.save(
                    save_obj, os.path.join(args.output_dir, f"{run_name}_{epoch}.pth")
                )

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            # dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    with torch.no_grad():
        torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/train_bev_drivelm.yaml")
    parser.add_argument("--output_dir", default="output/BEV_VQA_DriveLM")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--distributed", default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))

    main(args, config)
