import argparse


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="ViT-B-32")
    args.add_argument("--pretrained", type=str, default=None)
    args.add_argument("--precision", type=str, default="bf16")

    args.add_argument("--data_dir", type=str, default="")
    args.add_argument("--val_data_dir", type=str, default="")
    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--negtype", type=str, default="llm")

    args.add_argument("--device", type=str, default="cpu")

    args.add_argument("--train", action="store_true")
    args.add_argument("--epochs", type=int, default=92)
    args.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )

    args.add_argument("--lr", type=float, default=0.0005)
    args.add_argument("--wd", type=float, default=0.5)
    args.add_argument("--beta1", type=float, default=0.9)
    args.add_argument("--beta2", type=float, default=0.999)
    args.add_argument("--eps", type=str, default=1e-08)
    args.add_argument("--warmup", type=int, default=1000)

    args.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    args.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    args.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )

    args.add_argument("--ckpt", type=str, default=None)
    args.add_argument("--save_freq", type=int, default=5)
    args.add_argument("--log_dir", type=str, default=".")

    args.add_argument("--seed", type=int, default=42)

    args.add_argument("--wandb_resume", type=str, default=None)
    args.add_argument("--wandb_id", type=str, default=None)

    args = args.parse_args()
    return args
