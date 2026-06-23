import os
import json
import numpy as np
import os.path as op
from typing import List
from utils.iotools import read_json
from .bases import BaseDataset

class COCO(BaseDataset):
    """
    COCO Dataset compatible with cross-modal retrieval frameworks.
    """
    dataset_dir = 'COCO'

    def __init__(self, root='', verbose=True, test_mode='1k'):
        """
        Args:
            root (str): Root directory for data.
            verbose (bool): Whether to print dataset info.
            test_mode (str): '1k' or '5k'. 
                             '1k' loads json_data/COCO_test.json
                             '5k' loads json_data/COCO_testall.json
        """
        super(COCO, self).__init__()
        
        # 请修改为你实际的 COCO 图片数据根目录
        self.dataset_dir = "/home/yj/eangine/data/NCL/COCO_images"
        self.test_mode = test_mode
        
        # 1. 根据 test_mode 选择对应的测试文件
        if test_mode == '1k':
            test_json_path = 'json_data/COCO_test.json'
        elif test_mode == '5k':
            test_json_path = 'json_data/COCO_testall.json'
        else:
            raise ValueError(f"test_mode must be '1k' or '5k', but got {test_mode}")

        # 2. 加载数据
        with open('json_data/COCO_train.json', 'r', encoding='utf8') as fp_train:
            df_train = json.load(fp_train)
        with open('json_data/COCO_val.json', 'r', encoding='utf8') as fp_val:
            df_val = json.load(fp_val)
        
        # 加载动态选择的测试文件
        with open(test_json_path, 'r', encoding='utf8') as fp_test:
            df_test = json.load(fp_test)

        self.train_annos = df_train["caption"]
        self.test_annos  = df_test["caption"]
        self.val_annos   = df_val["caption"]

        # 3. 处理数据
        self.train, self.train_id_container = self._process_anno(df_train["caption"], df_train["image"], 'Train')
        self.test, self.test_id_container = self._process_anno(df_test["caption"], df_test["image"], 'Test')
        self.val, self.val_id_container = self._process_anno(df_val["caption"], df_val["image"], 'Val')

        if verbose:
            self.logger.info(f"=> MS-COCO Images and Captions (Mode: {self.test_mode}) are loaded.")
            self.logger.info(f"=> Loaded Test File: {test_json_path}")
            self.show_dataset_info()

    def _process_anno(self, annos, imgs, mode):
        pid_container = set()
        
        # ---------------------------
        # Train Mode
        # ---------------------------
        if mode == 'Train':
            dataset = []
            total_samples = len(annos)
            pid = (np.arange(total_samples) // 5).tolist()
            for i in range(len(annos)):
                if ("train" in imgs[i]) and ("val" in imgs[i]):
                    img_filename = "/COCO_val_ims" + imgs[i][30:]
                else:
                    img_filename = imgs[i][15:]
                
                dataset.append((pid[i], pid[i], self.dataset_dir+img_filename, annos[i]))
                pid_container.add(pid[i])
            
            for idx, pid in enumerate(pid_container):
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container

        # ---------------------------
        # Test & Val Mode
        # ---------------------------
        elif mode == 'Test' or mode == 'Val':
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []

            total_samples_annos = len(annos)
            total_samples_imgs  = len(imgs)

            assert total_samples_annos % 5 == 0, "COCO captions should be 5 per image"

            pid_caption = (np.arange(total_samples_annos) // 5).tolist()

            for i in range(total_samples_annos):
                img_idx = i // 5

                if i % 5 == 0:
                    current_pid = pid_caption[i]
                    pid_container.add(current_pid)

                    if mode == 'Test':
                        img_filename = "/COCO_val_ims/" + imgs[img_idx][30:]
                    else:
                        img_filename = "/COCO_val_ims/" + imgs[img_idx][29:]
                    
                    img_paths.append(self.dataset_dir+img_filename)
                    image_pids.append(current_pid)

                captions.append(annos[i])
                caption_pids.append(pid_caption[i])

            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
        return dataset, pid_container
