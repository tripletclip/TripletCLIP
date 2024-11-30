import os
import open_clip
import wandb
import logging

import torch
from torch import optim
import numpy as np
import random

from data import get_dataloader
from parse_args import get_args
from loss import tripletclip_loss, clip_loss
from train import train
from evaluate import evaluate
from logger import setup_logging

from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def main(args):
    # Set up logging
    args.log_path = os.path.join(args.log_dir, "out.log")
    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(args.log_path, logging.INFO)

    # Logging args
    logging.info(args)
    random_seed(args.seed, 0)
    
    # Load model
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model, _, transform = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        precision=args.precision
    )
    model.train()
    logging.info(f"Initialised model {args.model_name}")

    # Load dataset
    train_dataloader, val_dataloader = get_dataloader(
        args.data_dir, transform, tokenizer, args.negtype, args.train, args.batch_size, args.val_data_dir
    )
    logging.info(f"Initialised dataloaders")

    # Load optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.wd,
    )

    start_epoch = 0
    if args.ckpt is not None:
        logging.info(f"Loading checkpoint from {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

    scheduler = None
    if optimizer is not None:
        total_steps = len(train_dataloader) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = len(train_dataloader) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)
    logging.info(f"Initialised optimizer")

    criterion = tripletclip_loss
    clip_criterion = clip_loss

    # Initialise wandb
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="TripletCLIP",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"open_clip_tripletclip",
        # Track hyperparameters and run metadata
        config=vars(args),
        resume=args.wandb_resume,
        id=args.wandb_id,
    )
    wandb.watch(model, log="all", log_freq=10)

    if args.train:
        logging.info(f"Start Training!")
        train(
            args,
            model,
            train_dataloader,
            optimizer,
            criterion,
            clip_criterion,
            scheduler,
            start_epoch,
            args.device,
            val_dataloader,
        )
    else:
        logging.info(f"Evaluating")
        evaluate(args, model, val_dataloader, start_epoch, criterion, args.device)


if __name__ == "__main__":
    args = get_args()
    main(args)
