from torch.utils.data import Dataset
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import numpy as np
import os



def inject_noisy_correspondence(dataset, noisy_rate, noisy_file =None):
    logger = logging.getLogger("RASC.dataset")
    nums = len(dataset)
    dataset_copy = dataset.copy()
    captions  = [i[3] for i in dataset_copy]
    images    = [i[2] for i in dataset_copy]
    image_ids = [i[1] for i in dataset_copy]
    pids      = [i[0] for i in dataset_copy]

    noisy_inx = np.arange(nums)
    if noisy_rate > 0:
        print(noisy_file)
        random.seed(123)
        if os.path.exists(noisy_file):
            logger.info('=> Load noisy index from {}'.format(noisy_file))
            noisy_inx = np.load(noisy_file)  # 在正常的idx中添加噪声，{0,1,2,3...} ——> {0,100,2,78...}，100与78为错误对应
        else:
            inx = np.arange(nums)
            np.random.shuffle(inx)
            c_noisy_inx = inx[0: int(noisy_rate * nums)]
            shuffle_noisy_inx = np.array(c_noisy_inx)
            np.random.shuffle(shuffle_noisy_inx)
            noisy_inx[c_noisy_inx] = shuffle_noisy_inx
            np.save(noisy_file, noisy_inx)

    real_correspondeces = []

    for i in range(nums):
        if not os.path.exists(noisy_file):
            if noisy_inx[i]== i:
                real_correspondeces.append(1)
            else:
                real_correspondeces.append(0)
        else:
            if noisy_inx[i]== pids[i]:
                real_correspondeces.append(1)
            else:
                real_correspondeces.append(0)
        
        # pid, real_pid, image_id, image_path, text
        if noisy_rate > 0:
            tmp = (pids[i],image_ids[i],images[i],captions[noisy_inx[i]*5 + i%5])
        else:
            tmp = (pids[i],image_ids[i],images[i],captions[noisy_inx[i]])
        
        dataset_copy[i] = tmp
    logger.info(real_correspondeces[0:10])
    logger.info('=>Noisy rate: {},  clean pairs: {}, noisy pairs: {}, total pairs: {}'.format(noisy_rate, np.sum(real_correspondeces),nums-np.sum(real_correspondeces), nums))

    return dataset_copy, np.array(real_correspondeces)



class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("RASC.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


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
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
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
        img = read_image(img_path)
        
 
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
  
    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]
        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,args,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.txt_aug = args.txt_aug
        self.img_aug = args.img_aug
        if not "CC120K" in self.dataset[0][2]:
            self.dataset, self.real_correspondences = inject_noisy_correspondence(dataset,args.noisy_rate,args.noisy_file)
        else:
            self.real_correspondences = np.ones(len(self.dataset), dtype=np.int64)
            
        self.return_images_orig = getattr(args, "return_images_orig", False)
        self.return_raw_caption = getattr(args, "return_raw_caption", False) or getattr(args, "case_vis_enable", False)
        self.return_img_path = getattr(args, "return_img_path", False) or getattr(args, "case_vis_enable", False)

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            tensor_masked = self.transform(img)

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        
        if self.txt_aug:
            mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy().copy())
        
            ret = {
                'pids': pid,
                'image_ids': image_id,
                'images': tensor_masked,
                'caption_ids': caption_tokens,
                'index': index,
                'mlm_ids': mlm_tokens,
                'mlm_labels': mlm_labels,
                'real_label': int(self.real_correspondences[index])
            }
        else:
            ret = {
                'pids': pid,
                'image_ids': image_id,
                'images': tensor_masked,
                'caption_ids': caption_tokens,
                'index': index,
                'real_label': int(self.real_correspondences[index])
            }

        if self.return_images_orig:
            if hasattr(self, 'clean_transform') and self.clean_transform is not None:
                ret['images_orig'] = self.clean_transform(img)
            else:
                ret['images_orig'] = tensor_masked

        if self.return_img_path:
            ret['img_path'] = img_path

        if self.return_raw_caption:
            ret['raw_caption'] = caption

        return ret
    

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)


