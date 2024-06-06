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
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from models.blip_bev_pretrain import BLIP_BEV_Pretrain
import utils
from utils import warmup_lr_schedule, step_lr_schedule
from data import create_dataset, create_sampler, create_loader
from eval_blip_bev import LanguageEvaluation  # , GPTEvaluation


def train(
    model, data_loader, optimizer, epoch, device, config, writer, gen_log, gen_freq
):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "loss_ita", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_itm", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_lm", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 50
    num_samples = len(data_loader)

    data_loader.sampler.set_epoch(epoch)

    for i, (bev, statement) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if epoch == 1:
            warmup_lr_schedule(
                optimizer,
                i,
                config["warmup_steps"],
                config["warmup_lr"],
                config["init_lr"],
            )

        optimizer.zero_grad()

        bev = bev.to(device, non_blocking=True)

        # ramp up alpha in the first 2 epochs
        alpha = config["alpha"] * min(
            1, (epoch * len(data_loader) + i) / (2 * len(data_loader))
        )

        loss_ita, loss_itm, loss_lm = model(bev, statement, alpha=alpha)
        loss = loss_ita + loss_itm + loss_lm

        loss.backward()
        optimizer.step()

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_lm=loss_lm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        global_step = epoch * len(data_loader) + i

        writer.add_scalar("Train/Loss", loss.item(), global_step)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("Train/ITA", loss_ita.item(), global_step)
        writer.add_scalar("Train/ITM", loss_itm.item(), global_step)
        writer.add_scalar("Train/LM", loss_lm.item(), global_step)

        if i % gen_freq == 0:
            generate_log_entry(model, bev, statement, epoch, i, gen_log, mode="Train")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {
        k: "{:.3f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }


def validation(model, data_loader, epoch, device, config, writer, gen_log, gen_freq):
    print(f"\n[EPOCH: {epoch}] Starting validation...\n")
    ptb_tokenizer = PTBTokenizer()
    model.eval()
    with torch.no_grad():
        lang_metrics = {
            "Bleu_1": 0.0,
            "Bleu_2": 0.0,
            "Bleu_3": 0.0,
            "Bleu_4": 0.0,
            # "METEOR": 0.0,
            "ROUGE_L": 0.0,
            "CIDEr": 0.0,
            # "SPICE": 0.0,
        }

        gpt_metric = 0.0
        num_gpt_fail = 0
        num_lang_fail = 0
        num_samples = len(data_loader)

        for i, (bev, statement) in enumerate(data_loader):
            print(f"\rRunning Validation {i + 1}/{num_samples}", end="")
            bev = bev.to(device, non_blocking=True)
            output = model.generate(bev)

            # LANGUAGE METRICS CALCULATION
            try:
                lang_scores = LanguageEvaluation.evaluate(list(output), list(statement))
                lang_metrics["Bleu_1"] += float(lang_scores["Bleu_1"])
                lang_metrics["Bleu_2"] += float(lang_scores["Bleu_2"])
                lang_metrics["Bleu_3"] += float(lang_scores["Bleu_3"])
                lang_metrics["Bleu_4"] += float(lang_scores["Bleu_4"])
                # lang_metrics["METEOR"] += float(lang_scores["METEOR"])
                lang_metrics["ROUGE_L"] += float(lang_scores["ROUGE_L"])
                lang_metrics["CIDEr"] += float(lang_scores["CIDEr"])
                # lang_metrics["SPICE"] += float(lang_scores["SPICE"])
            except:
                num_lang_fail += 1

            """
            # GPT METRIC CALCULATION
            try:
                gpt_score = float(GPTEvaluation.evaluate(output[0], statement[0]))
            except:
                gpt_score = 0
                num_gpt_fail += 1
            
            gpt_metric += gpt_score
            """

            if i % gen_freq == 0:
                generate_log_entry(model, bev, statement, epoch, i, gen_log, mode="Val")

        # Averaging over epoch
        for m in lang_metrics:
            lang_metrics[m] = lang_metrics[m] / max(1, num_samples - num_lang_fail)

        # gpt_metric = gpt_metric / max(1, num_samples- num_gpt_fail)

        # writer.add_scalar("Val/GPT", gpt_metric, epoch)
        writer.add_scalar("Val/BLEU-1", lang_metrics["Bleu_1"], epoch)
        writer.add_scalar("Val/BLEU-2", lang_metrics["Bleu_2"], epoch)
        writer.add_scalar("Val/BLEU-3", lang_metrics["Bleu_3"], epoch)
        writer.add_scalar("Val/BLEU-4", lang_metrics["Bleu_4"], epoch)
        # writer.add_scalar("Val/METEOR", lang_metrics["METEOR"], epoch)
        writer.add_scalar("Val/ROUGE-L", lang_metrics["ROUGE_L"], epoch)
        writer.add_scalar("Val/CIDEr", lang_metrics["CIDEr"], epoch)
        # writer.add_scalar("Val/SPICE", lang_metrics["SPICE"], epoch)

        # print("GPT metric fails during validation:", num_gpt_fail)
        print("\nLanguage metric fails during validation:", num_lang_fail)
        print(f"\n[EPOCH: {epoch}] Validation complete!\n")

    model.train()


def generate_log_entry(model, bev, statement, ep, step, log_file, mode="Train"):
    output = model.generate(bev)
    print(f"\nEpoch: {ep}, Mode: {mode}, Step: {step}", file=log_file, flush=True)
    print(f"GT:  {statement[0]}", file=log_file, flush=True)
    print(f"Out: {output[0]}", file=log_file, flush=True)


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating datasets")
    train_dataset, val_dataset = create_dataset("pretrain_bev", config, min_scale=0.2)
    print("number of training samples: %d" % len(train_dataset))
    print("number of validation samples: %d" % len(val_dataset))

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
    model = BLIP_BEV_Pretrain(queue_size=config["queue_size"])
    model = model.to(device)
    print("Model initialized.")

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config["init_lr"],
        weight_decay=config["weight_decay"],
    )

    run_name = "BEV_Pretrain"
    todays_date = datetime.now().strftime("%d-%m")
    sum_writer = SummaryWriter(log_dir=f"runs/{todays_date}_{run_name}")

    start_epoch = 1

    # CONTINUE FROM CHECKPOINT ----------------------------
    """print("Loading previous checkpoint...")
    checkpoint = torch.load(r"")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    print("Previous checkpoint loaded!")"""
    # ----------------------------------------------------

    """
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module 
    """

    with open(f"./logs/log_{todays_date}_{run_name}.txt", "w") as gen_log_file:
        validation(model, val_loader, 0, device, config, sum_writer, gen_log_file, gen_freq=100)

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
                model,
                val_loader,
                epoch,
                device,
                config,
                sum_writer,
                gen_log_file,
                gen_freq=100,
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

            dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    with torch.no_grad():
        torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/pretrain_bev.yaml")
    parser.add_argument("--output_dir", default="output/Pretrain_BEV")
    # parser.add_argument('--checkpoint', default='')
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
