import os

import numpy as np

from .bases import BaseDataset


class CC120K(BaseDataset):
    dataset_dir = "CC120K"

    def __init__(self, root="", verbose=True):
        super(CC120K, self).__init__()
        self.dataset_dir = os.path.join(root, self.dataset_dir) if root else self.dataset_dir
        self.image_root = os.path.join(self.dataset_dir, "images")
        self.anno_root = os.path.join(self.dataset_dir, "annotations", "scan_split")

        train_ids = self._load_txt(os.path.join(self.anno_root, "train_ids.txt"))
        train_caps = self._load_txt(os.path.join(self.anno_root, "train_caps.txt"))

        dev_ids_path = os.path.join(self.anno_root, "dev_ids.txt")
        if os.path.exists(dev_ids_path):
            val_ids = self._load_txt(dev_ids_path)
            val_caps = self._load_txt(os.path.join(self.anno_root, "dev_caps.txt"))
        else:
            val_ids, val_caps = [], []

        test_ids_path = os.path.join(self.anno_root, "test_ids.txt")
        if os.path.exists(test_ids_path):
            test_ids = self._load_txt(test_ids_path)
            test_caps = self._load_txt(os.path.join(self.anno_root, "test_caps.txt"))
        else:
            test_ids, test_caps = [], []

        self.train_annos = train_caps
        self.test_annos = test_caps
        self.val_annos = val_caps

        self.train, self.train_id_container = self._process_anno(train_caps, train_ids, "Train")
        self.val, self.val_id_container = self._process_anno(val_caps, val_ids, "Val")
        self.test, self.test_id_container = self._process_anno(test_caps, test_ids, "Test")

        if verbose:
            self.logger.info("=> Conceptual Captions are loaded")
            self.show_dataset_info()

    def _load_txt(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]

    def _image_path(self, image_id):
        return os.path.join(self.image_root, str(image_id) + ".jpg")

    def _process_anno(self, annos, ids, mode):
        assert len(annos) == len(ids), f"Error: {mode} set has {len(ids)} ids but {len(annos)} captions"
        pid_container = set()

        if mode == "Train":
            dataset = []
            pids = np.arange(len(annos)).tolist()
            for i, caption in enumerate(annos):
                dataset.append((pids[i], pids[i], self._image_path(ids[i]), caption))
                pid_container.add(pids[i])
            return dataset, pid_container

        dataset = {"image_pids": [], "img_paths": [], "caption_pids": [], "captions": []}
        pids = np.arange(len(annos)).tolist()
        for i, caption in enumerate(annos):
            dataset["img_paths"].append(self._image_path(ids[i]))
            dataset["captions"].append(caption)
            dataset["image_pids"].append(pids[i])
            dataset["caption_pids"].append(pids[i])
            pid_container.add(pids[i])
        return dataset, pid_container