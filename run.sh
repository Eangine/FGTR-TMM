#!/usr/bin/env bash
set -e

# Flickr30k, 40% noise, HHNC default setting.
python train.py \
  --img_aug \
  --loss_names InfoNCE \
  --warm_up \
  --memory_size 40000 \
  --dataset_name Flickr30k \
  --warmup_epochs 5 \
  --num_epoch 10 \
  --lr 1e-6 \
  --noisy_rate 0.4 \
  --noisy_file ./noise_file/f30k/noise_inx_0.4.npy

# COCO examples:
# python train.py --img_aug --loss_names InfoNCE --warm_up --memory_size 40000 --dataset_name COCO --warmup_epochs 5 --num_epoch 10 --lr 1e-6 --noisy_rate 0.2 --noisy_file ./noise_file/coco/noise_inx_0.2.npy
# python train.py --img_aug --loss_names InfoNCE --warm_up --memory_size 40000 --dataset_name COCO --warmup_epochs 5 --num_epoch 10 --lr 1e-6 --noisy_rate 0.4 --noisy_file ./noise_file/coco/noise_inx_0.4.npy
# python train.py --img_aug --loss_names InfoNCE --warm_up --memory_size 40000 --dataset_name COCO --warmup_epochs 5 --num_epoch 10 --lr 1e-6 --noisy_rate 0.6 --noisy_file ./noise_file/coco/noise_inx_0.6.npy

# Evaluation example:
# python test.py --config_file HHNC_Result/Flickr30k/<run_name>/configs.yaml --checkpoint best.pth