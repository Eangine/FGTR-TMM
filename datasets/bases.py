from torch.utils.data import Dataset
import logging
import os
import random

import numpy as np
import torch
from prettytable import PrettyTable

from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer


def inject_noisy_correspondence(dataset, noisy_rate, noisy_file=None):
    logger = logging.getLogger("HHNC.dataset")
    nums = len(dataset)
    dataset_copy = dataset.copy()
    captions = [item[3] for item in dataset_copy]
    images = [item[2] for item in dataset_copy]
    image_ids = [item[1] for item in dataset_copy]
    pids = [item[0] for item in dataset_copy]

    noisy_index = np.asarray(pids, dtype=np.int64)
    loaded_from_file = bool(noisy_file and os.path.exists(noisy_file))
    if noisy_rate > 0:
        logger.info(f"noise file: {noisy_file}")
        random.seed(123)
        if loaded_from_file:
            logger.info(f"=> Load noisy index from {noisy_file}")
            noisy_index = np.load(noisy_file)
        else:
            index = np.arange(nums)
            np.random.shuffle(index)
            noisy_samples = index[: int(noisy_rate * nums)]
            shuffled_pids = noisy_index[noisy_samples].copy()
            np.random.shuffle(shuffled_pids)
            noisy_index[noisy_samples] = shuffled_pids
            if noisy_file:
                os.makedirs(os.path.dirname(noisy_file), exist_ok=True)
                np.save(noisy_file, noisy_index)

    real_correspondences = []
    for i in range(nums):
        real_correspondences.append(1 if noisy_rate <= 0 or noisy_index[i] == pids[i] else 0)

        if noisy_rate > 0:
            caption = captions[noisy_index[i] * 5 + i % 5]
        else:
            caption = captions[i]
        dataset_copy[i] = (pids[i], image_ids[i], images[i], caption)

    real_correspondences = np.asarray(real_correspondences)
    logger.info(real_correspondences[:10].tolist())
    logger.info(
        "=> Noisy rate: {}, clean pairs: {}, noisy pairs: {}, total pairs: {}".format(
            noisy_rate,
            int(real_correspondences.sum()),
            nums - int(real_correspondences.sum()),
            nums,
        )
    )
    return dataset_copy, real_correspondences


class BaseDataset(object):
    logger = logging.getLogger("HHNC.dataset")

    def show_dataset_info(self):
        num_train_pids = len(self.train_id_container)
        num_train_imgs = len(self.train_annos)
        num_train_captions = len(self.train)
        num_test_pids = len(self.test_id_container)
        num_test_imgs = len(self.test_annos)
        num_test_captions = len(self.test["captions"])
        num_val_pids = len(self.val_id_container)
        num_val_imgs = len(self.val_annos)
        num_val_captions = len(self.val["captions"])

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(["subset", "ids", "images", "captions"])
        table.add_row(["train", num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(["test", num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(["val", num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info("\n" + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(f"Input {caption} is too long for context length {text_length}")
    result[: len(tokens)] = torch.tensor(tokens)
    return result


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        image = read_image(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return pid, image


class TextDataset(Dataset):
    def __init__(self, caption_pids, captions, text_length=77, truncate=True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]
        caption = tokenize(
            caption,
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=self.truncate,
        )
        return pid, caption


class ImageTextDataset(Dataset):
    def __init__(self, dataset, args, transform=None, text_length=77, truncate=True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.img_aug = args.img_aug

        if "CC120K" not in self.dataset[0][2]:
            self.dataset, self.real_correspondences = inject_noisy_correspondence(
                dataset,
                args.noisy_rate,
                args.noisy_file,
            )
        else:
            self.real_correspondences = np.ones(len(self.dataset), dtype=np.int64)

        self.return_images_orig = getattr(args, "return_images_orig", False)
        self.return_raw_caption = getattr(args, "return_raw_caption", False)
        self.return_img_path = getattr(args, "return_img_path", False)
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        image = read_image(img_path)
        if self.transform is not None:
            image = self.transform(image)

        ret = {
            "pids": pid,
            "image_ids": image_id,
            "images": image,
            "caption_ids": tokenize(
                caption,
                tokenizer=self.tokenizer,
                text_length=self.text_length,
                truncate=self.truncate,
            ),
            "index": index,
            "real_label": int(self.real_correspondences[index]),
        }

        if self.return_images_orig:
            ret["images_orig"] = image
        if self.return_img_path:
            ret["img_path"] = img_path
        if self.return_raw_caption:
            ret["raw_caption"] = caption

        return ret