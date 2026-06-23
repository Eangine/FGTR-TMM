import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset

import os
import json
import numpy as np

class Flickr30k(BaseDataset):

    dataset_dir = 'Flickr30k'

    def __init__(self, root='', verbose=True):
        super(Flickr30k, self).__init__()
        # self.dataset_dir = op.join(root, self.dataset_dir)
        self.dataset_dir = "/home/yj/eangine/data/NCL/flickr30k-images/"

        with open('json_data/f30k_train.json', 'r', encoding='utf8') as fp_train:
            df_train = json.load(fp_train)
        with open('json_data/f30k_val.json', 'r', encoding='utf8') as fp_val:
            df_val = json.load(fp_val)
        with open('json_data/f30k_test.json', 'r', encoding='utf8') as fp_test:
            df_test = json.load(fp_test)

        self.train_annos = df_train["caption"]
        self.test_annos  = df_test["caption"]
        self.val_annos   = df_val["caption"]

        self.train, self.train_id_container = self._process_anno(df_train["caption"], df_train["image"], 'Train')
        self.test, self.test_id_container = self._process_anno(df_test["caption"], df_test["image"], 'Test')
        self.val, self.val_id_container = self._process_anno(df_val["caption"], df_val["image"], 'Val')

        if verbose:
            self.logger.info("=> Flickr30k Images and Captions are loaded")
            self.show_dataset_info()
    
    def _process_anno(self, annos, imgs, mode):
        pid_container = set()
        if mode == 'Train':
            dataset = []
            total_samples = len(annos)
            pid = (np.arange(total_samples) // 5).tolist()
            for i in range(len(annos)):
                dataset.append((pid[i], pid[i], os.path.join(self.dataset_dir,imgs[i][41:]), annos[i]))
                pid_container.add(pid[i])
            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        elif mode == 'Test':
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            total_samples_img = len(imgs)
            total_samples_annos = len(annos)
            pid = (np.arange(total_samples_img)).tolist()
            pid_caption = (np.arange(total_samples_annos) // 5).tolist()
            for i in range(total_samples_annos):
                if i%5==0:
                    pid_container.add(pid[i//5])
                    img_paths.append(os.path.join(self.dataset_dir, imgs[i//5][40:]))
                    image_pids.append(pid[i//5])
                captions.append(annos[i])
                caption_pids.append(pid_caption[i])

            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            total_samples_img = len(imgs)
            total_samples_annos = len(annos)
            pid = (np.arange(total_samples_img)).tolist()
            pid_caption = (np.arange(total_samples_annos) // 5).tolist()
            for i in range(total_samples_annos):
                if i%5==0:
                    pid_container.add(pid[i//5])
                    img_paths.append(os.path.join(self.dataset_dir, imgs[i//5][39:]))
                    image_pids.append(pid[i//5])
                captions.append(annos[i])
                caption_pids.append(pid_caption[i])

            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container



    

