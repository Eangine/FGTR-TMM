import os.path as op

import numpy as np

from .bases import BaseDataset
from utils.iotools import read_json


class Flickr30k(BaseDataset):
    dataset_dir = "flickr30k_images"

    def __init__(self, root="", verbose=True):
        super(Flickr30k, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir) if root else self.dataset_dir

        df_train = read_json("json_data/f30k_train.json")
        df_val = read_json("json_data/f30k_val.json")
        df_test = read_json("json_data/f30k_test.json")

        self.train_annos = df_train["caption"]
        self.test_annos = df_test["caption"]
        self.val_annos = df_val["caption"]

        self.train, self.train_id_container = self._process_anno(df_train["caption"], df_train["image"], "Train")
        self.test, self.test_id_container = self._process_anno(df_test["caption"], df_test["image"], "Test")
        self.val, self.val_id_container = self._process_anno(df_val["caption"], df_val["image"], "Val")

        if verbose:
            self.logger.info("=> Flickr30k images and captions are loaded")
            self.show_dataset_info()

    def _resolve_image_path(self, raw_path):
        normalized = raw_path.replace("\\", "/")
        for marker in ("flickr30k_images/", "flickr30k-images/"):
            if marker in normalized:
                rel_path = normalized.split(marker, 1)[1]
                return op.join(self.dataset_dir, *rel_path.split("/"))
        return op.join(self.dataset_dir, op.basename(normalized))

    def _process_anno(self, annos, imgs, mode):
        pid_container = set()
        if mode == "Train":
            dataset = []
            pids = (np.arange(len(annos)) // 5).tolist()
            for i, caption in enumerate(annos):
                dataset.append((pids[i], pids[i], self._resolve_image_path(imgs[i]), caption))
                pid_container.add(pids[i])
            for idx, pid in enumerate(pid_container):
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container

        dataset = {"image_pids": [], "img_paths": [], "caption_pids": [], "captions": []}
        total_samples_annos = len(annos)
        caption_pids = (np.arange(total_samples_annos) // 5).tolist()
        for i, caption in enumerate(annos):
            if i % 5 == 0:
                image_pid = i // 5
                pid_container.add(image_pid)
                dataset["img_paths"].append(self._resolve_image_path(imgs[image_pid]))
                dataset["image_pids"].append(image_pid)
            dataset["captions"].append(caption)
            dataset["caption_pids"].append(caption_pids[i])
        return dataset, pid_container