import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import os.path as op
import random
import time
import warnings

import numpy as np
import torch

from datasets import build_dataloader
from model.build import build_model
from processor.processor import do_inference, do_train
from solver import build_lr_scheduler, build_optimizer
from utils.checkpoint import Checkpointer
from utils.comm import get_rank, synchronize
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from utils.metrics import Evaluator
from utils.options import get_args

warnings.filterwarnings("ignore")


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    args = get_args()
    set_seed(1 + get_rank())

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f"{cur_time}_{args.name}_{args.loss_names}")

    logger = setup_logger("HHNC", save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(",", "\n"))

    save_train_configs(args.output_dir, args)
    os.makedirs(op.join(args.output_dir, "img"), exist_ok=True)

    train_loader, val_img_loader, val_txt_loader, _ = build_dataloader(args)
    model = build_model(args)
    logger.info("Total params: %2.fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    if getattr(args, "profile_complexity", False):
        from utils.complexity import profile_model_complexity

        profile_model_complexity(
            model=model,
            args=args,
            save_dir=args.output_dir,
            device=device,
            batch_size=getattr(args, "profile_batch_size", 64),
            warmup=getattr(args, "profile_warmup", 20),
            iters=getattr(args, "profile_iters", 100),
        )
        raise SystemExit(0)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    optimizer = build_optimizer(args, model, mode="train")
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint["epoch"]
        logger.info(f"===================> start {start_epoch}")

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)

    logger.info("===================> start test")
    args.training = False
    test_img_loader, test_txt_loader, _ = build_dataloader(args)

    for checkpoint_name in ("best.pth", "last.pth"):
        checkpoint_path = op.join(args.output_dir, checkpoint_name)
        if not op.exists(checkpoint_path):
            continue
        model = build_model(args)
        checkpointer = Checkpointer(model)
        checkpointer.load(f=checkpoint_path)
        model = model.cuda()
        logger.info(f"Testing {checkpoint_name}")
        do_inference(model, test_img_loader, test_txt_loader)