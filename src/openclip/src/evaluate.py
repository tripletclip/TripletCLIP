import torch

from tqdm import tqdm
import wandb


def evaluate(args, model, val_dataloader, epoch, criterion, device):
    epoch_bar = tqdm(
        val_dataloader, desc=f"Val Epoch {epoch}/{args.epochs}", leave=False
    )
    running_loss = 0.0
    running_accuracy = 0.0  # Add this to track accuracy
    for batch_idx, (image, input_ids) in enumerate(epoch_bar):
        image = image.to(device, dtype=torch.bfloat16)
        input_ids = input_ids.to(device, dtype=torch.long)
        with torch.no_grad(), torch.amp.autocast(
            device_type=device, dtype=torch.bfloat16
        ):
            img_embs, text_embs, logit_scale = model(image, input_ids)
            
            loss, accuracy = criterion(img_embs, text_embs, logit_scale)

        running_loss += loss.item()
        running_accuracy += accuracy  # Accumulate accuracy

        epoch_bar.set_postfix(
            loss=running_loss / (batch_idx + 1),
            accuracy=running_accuracy / (batch_idx + 1),
        )

        # Log loss and accuracy at each step (each batch) to Wandb
        wandb.log(
            {
                "val_step_loss": loss.item(),
                "val_step_accuracy": accuracy,
                "val_epoch": epoch,
                "val_batch_idx": batch_idx,
            }
        )

    epoch_loss = running_loss / len(val_dataloader)
    epoch_accuracy = running_accuracy / len(val_dataloader)

    # Log epoch-level loss and accuracy to Wandb
    wandb.log(
        {
            "val_epoch_loss": epoch_loss,
            "val_epoch_accuracy": epoch_accuracy,
            "val_epoch": epoch,
        }
    )

    epoch_bar.clear()
    tqdm.write(
        f"Val Epoch {epoch}/{args.epochs} completed, val average loss: {epoch_loss:.4f}, val average accuracy: {epoch_accuracy:.4f}"
    )
