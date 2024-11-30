import os
from tqdm import tqdm
import wandb

import torch

from evaluate import evaluate


def train_one_epoch(
    args, model, epoch, dataloader, optimizer, criterion, scheduler=None, device="cuda"
):
    epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
    running_loss = 0.0
    running_accuracy = 0.0  # Add this to track accuracy
    for batch_idx, (image, neg_image, input_ids, neg_input_ids) in enumerate(epoch_bar):
        image = image.to(device, dtype=torch.bfloat16)
        neg_image = neg_image.to(device, dtype=torch.bfloat16)
        input_ids = input_ids.to(device, dtype=torch.long)
        neg_input_ids = neg_input_ids.to(device, dtype=torch.long)
        with torch.enable_grad(), torch.amp.autocast(
            device_type=device, dtype=torch.bfloat16
        ):
            img_embs, text_embs, logit_scale = model(image, input_ids)
            neg_img_embs, neg_text_embs, _ = model(neg_image, neg_input_ids)

            loss, accuracy = criterion(img_embs, text_embs, neg_img_embs, neg_text_embs, logit_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                step = epoch * len(dataloader) + batch_idx
                scheduler(step)

        running_loss += loss.item()
        running_accuracy += accuracy  # Accumulate accuracy

        epoch_bar.set_postfix(
            loss=running_loss / (batch_idx + 1),
            accuracy=running_accuracy / (batch_idx + 1),
        )

        # Log loss and accuracy at each step (each batch) to Wandb
        wandb.log(
            {
                "step_loss": loss.item(),
                "step_accuracy": accuracy,
                "epoch": epoch,
                "batch_idx": batch_idx,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader)

    # Log epoch-level loss and accuracy to Wandb
    wandb.log(
        {"epoch_loss": epoch_loss, "epoch_accuracy": epoch_accuracy, "epoch": epoch}
    )

    epoch_bar.clear()
    tqdm.write(
        f"Epoch {epoch}/{args.epochs} completed, average loss: {epoch_loss:.4f}, average accuracy: {epoch_accuracy:.4f}"
    )

    if epoch % args.save_freq == 0:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }

        os.makedirs(f"{args.log_dir}/checkpoints", exist_ok=True)
        torch.save(checkpoint, f"{args.log_dir}/checkpoints/epoch-{epoch}.ckpt")


def train(
    args,
    model,
    dataloader,
    optimizer,
    criterion,
    clip_criterion,
    scheduler=None,
    start_epoch=0,
    device="cuda",
    val_dataloader=None,
):
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(
            args, model, epoch, dataloader, optimizer, criterion, scheduler, device
        )

        if val_dataloader is not None and epoch % args.save_freq  == 0:
            evaluate(args, model, val_dataloader, epoch, clip_criterion, device)

    wandb.finish()
