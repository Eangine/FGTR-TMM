
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler
from utils.comm import get_world_size
import open_clip

from .bases import ImageDataset, TextDataset, ImageTextDataset
from .flickr30k import Flickr30k
from .COCO import COCO
from .CC import CC120K

__factory = {'Flickr30k': Flickr30k, 'COCO': COCO, 'CC120K': CC120K}


def _build_openclip_transforms():
    """Build CLIP image transforms without instantiating the full model when possible."""
    if hasattr(open_clip, "get_model_config") and hasattr(open_clip, "image_transform"):
        model_cfg = open_clip.get_model_config("ViT-B-32")
        image_cfg = model_cfg.get("vision_cfg", {}) if model_cfg else {}
        image_size = image_cfg.get("image_size", 224)
        default_mean = getattr(open_clip, "OPENAI_DATASET_MEAN", (0.48145466, 0.4578275, 0.40821073))
        default_std = getattr(open_clip, "OPENAI_DATASET_STD", (0.26862954, 0.26130258, 0.27577711))
        mean = image_cfg.get("image_mean", default_mean)
        std = image_cfg.get("image_std", default_std)

        preprocess_train = open_clip.image_transform(
            image_size=image_size,
            is_train=True,
            mean=mean,
            std=std,
        )
        preprocess_val = open_clip.image_transform(
            image_size=image_size,
            is_train=False,
            mean=mean,
            std=std,
        )
        return preprocess_train, preprocess_val

    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='laion2b_s34b_b79k',
    )
    return preprocess_train, preprocess_val


def _loader_kwargs(num_workers, training=False):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = training
        kwargs["prefetch_factor"] = 1
    return kwargs

def collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if v[0] is None:
            continue
        if isinstance(v[0], int):
            batch_tensor_dict[k] = torch.tensor(v)
        elif torch.is_tensor(v[0]):
            batch_tensor_dict[k] = torch.stack(v)
        elif isinstance(v[0], str):
            batch_tensor_dict[k] = v
        elif isinstance(v[0], list):
            batch_tensor_dict[k] = v
        else:
            print(f"Warning: Unexpected data type {type(v[0])} for key {k} in collate_fn")
            
    return batch_tensor_dict

def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)
    
    logger.info("Building OpenCLIP image transforms...")
    preprocess_train, preprocess_val = _build_openclip_transforms()
    val_transforms = preprocess_val

    norm_mean = (0.48145466, 0.4578275, 0.40821073)
    for t in preprocess_val.transforms:
        if isinstance(t, T.Normalize):
            norm_mean = t.mean
            break

    if args.training:
        if args.img_aug:

            # train_transforms = preprocess_train
            # train_transforms.transforms.append(
            #     T.RandomErasing(scale=(0.02, 0.4), value=norm_mean)
            # )
            # logger.info("Added RandomErasing to OpenCLIP train transforms.")

            model_cfg = open_clip.get_model_config("ViT-B-32")
            image_cfg = model_cfg.get("vision_cfg", {}) if model_cfg else {}
            image_size = image_cfg.get("image_size", 224)
            default_mean = getattr(open_clip, "OPENAI_DATASET_MEAN", (0.48145466, 0.4578275, 0.40821073))
            default_std = getattr(open_clip, "OPENAI_DATASET_STD", (0.26862954, 0.26130258, 0.27577711))
            mean = image_cfg.get("image_mean", default_mean)
            std = image_cfg.get("image_std", default_std)
            train_transforms = T.Compose([
                T.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)], p=0.5),
                T.RandomApply([T.GaussianBlur(3) ], p=0.2), T.ToTensor(), T.Normalize(mean, std),
                T.RandomErasing(p=0.5, scale=(0.02, 0.4), value=norm_mean)
            ])
            
        else:
            train_transforms = preprocess_val
        
        train_set = ImageTextDataset(
            dataset.train, 
            args, 
            train_transforms, 
            text_length=args.text_length
        )

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                mini_batch_size = args.batch_size // get_world_size()
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)
                
                train_loader = DataLoader(train_set,
                                          batch_sampler=batch_sampler,
                                          collate_fn=collate,
                                          **_loader_kwargs(num_workers, training=True))
            else:
                logger.info(f'using random identity sampler: batch_size: {args.batch_size}')
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(dataset.train, args.batch_size, args.num_instance),
                                          collate_fn=collate,
                                          **_loader_kwargs(num_workers, training=True))
        elif args.sampler == 'random':
            logger.info('using random sampler')
            if args.distributed:
                sampler = DistributedSampler(train_set)
                shuffle = False
            else:
                sampler = None
                shuffle = True

            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=shuffle,
                                      sampler=sampler,
                                      collate_fn=collate,
                                      **_loader_kwargs(num_workers, training=True))
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'], val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'], ds['captions'], text_length=args.text_length)
        
        val_batch_size = args.batch_size
        if args.dataset_name == 'Flickr30k':
            val_batch_size = 507
        elif args.dataset_name == 'COCO':
            val_batch_size = 500

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=val_batch_size,
                                    shuffle=False,
                                    **_loader_kwargs(num_workers))
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=val_batch_size,
                                    shuffle=False,
                                    **_loader_kwargs(num_workers))

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # build dataloader for testing
        test_transforms = tranforms if tranforms else val_transforms

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'], test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'], ds['captions'], text_length=args.text_length)
        
        test_batch_size = args.test_batch_size
        if args.dataset_name == 'Flickr30k':
            test_batch_size = 500

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=test_batch_size,
                                     shuffle=False,
                                     **_loader_kwargs(num_workers))
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=test_batch_size,
                                     shuffle=False,
                                     **_loader_kwargs(num_workers))
        return test_img_loader, test_txt_loader, num_classes


