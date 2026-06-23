import os
import os.path as op
import numpy as np
from typing import List
from .bases import BaseDataset

class CC120K(BaseDataset):

    dataset_dir = 'CC120K' 

    def __init__(self, root='', verbose=True):
        super(CC120K, self).__init__()
        
        # 1. 设定新的根目录
        # 注意：这里直接指向你提供的新路径
        self.dataset_dir = "/home/yj/eangine/data/NCL/CC120K/"
        self.image_root = os.path.join(self.dataset_dir, 'images')
        self.anno_root = os.path.join(self.dataset_dir, 'annotations/scan_split')

        # 2. 读取 TXT 数据
        # 假设 CC 的验证集命名为 'dev'，测试集为 'test'，训练集为 'train'
        train_ids = self._load_txt(os.path.join(self.anno_root, 'train_ids.txt'))
        train_caps = self._load_txt(os.path.join(self.anno_root, 'train_caps.txt'))
        
        # 注意：CC 数据集有时没有标准的 test/dev split，如果没有文件，你需要确认文件名
        # 这里为了防止报错，加了简单的检查，如果文件不存在则设为空列表
        dev_ids_path = os.path.join(self.anno_root, 'dev_ids.txt')
        if os.path.exists(dev_ids_path):
            val_ids = self._load_txt(dev_ids_path)
            val_caps = self._load_txt(os.path.join(self.anno_root, 'dev_caps.txt'))
        else:
            val_ids, val_caps = [], []

        test_ids_path = os.path.join(self.anno_root, 'test_ids.txt')
        if os.path.exists(test_ids_path):
            test_ids = self._load_txt(test_ids_path)
            test_caps = self._load_txt(os.path.join(self.anno_root, 'test_caps.txt'))
        else:
            test_ids, test_caps = [], []

        # 3. 处理数据格式以适配 BaseDataset
        # 为了兼容性，我们将 caption 列表赋值给 self.*_annos
        self.train_annos = train_caps
        self.test_annos  = test_caps
        self.val_annos   = val_caps

        # 4. 调用处理函数生成最终数据列表
        self.train, self.train_id_container = self._process_anno(train_caps, train_ids, 'Train')
        self.val, self.val_id_container = self._process_anno(val_caps, val_ids, 'Val')
        self.test, self.test_id_container = self._process_anno(test_caps, test_ids, 'Test')

        if verbose:
            self.logger.info("=> Conceptual Captions loaded")
            self.show_dataset_info()

    def _load_txt(self, path):
        """辅助函数：按行读取 txt 文件"""
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                lines.append(line.strip())
        return lines

    def _process_anno(self, annos, ids, mode):
        pid_container = set()
        
        # 确保 id 和 caption 数量一致
        assert len(annos) == len(ids), f"Error: {mode} set has {len(ids)} ids but {len(annos)} captions"

        if mode == 'Train':
            dataset = []
            # CC 数据集通常是一对一，所以不需要 // 5
            # 生成简单的 0 到 N 的索引作为 pid
            pids = np.arange(len(annos)).tolist()
            
            for i in range(len(annos)):
                # 构建图片路径: root/images/id.jpg
                # 注意：ids[i] 可能是字符串，如果是 CC 数据集直接拼接 .jpg
                img_path = os.path.join(self.image_root, str(ids[i]) + '.jpg')
                
                dataset.append((pids[i], pids[i], img_path, annos[i]))
                pid_container.add(pids[i])

            return dataset, pid_container

        elif mode == 'Test' or mode == 'Val':
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            
            pids = np.arange(len(annos)).tolist()
            
            for i in range(len(annos)):
                path = os.path.join(self.image_root, str(ids[i]) + '.jpg')
                
                img_paths.append(path)
                captions.append(annos[i])
                
                image_pids.append(pids[i])
                caption_pids.append(pids[i])
                
                pid_container.add(pids[i])

            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container