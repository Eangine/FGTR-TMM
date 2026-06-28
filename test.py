import argparse
import os.path as op

from datasets import build_dataloader
from model.build import build_model
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HHNC evaluation")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", default="best.pth")
    cli_args = parser.parse_args()

    args = load_train_configs(cli_args.config_file)
    args.training = False

    logger = setup_logger("HHNC", save_dir=args.output_dir, if_train=args.training)
    logger.info(args)

    test_img_loader, test_txt_loader, _ = build_dataloader(args)

    checkpoint_path = cli_args.checkpoint
    if not op.isabs(checkpoint_path):
        checkpoint_path = op.join(args.output_dir, checkpoint_path)

    model = build_model(args)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=checkpoint_path)
    model = model.cuda()

    logger.info(f"Testing checkpoint: {checkpoint_path}")
    do_inference(model, test_img_loader, test_txt_loader)