import json
import os
import sys

import torch
import torch.nn.functional as F

from configs import Config
from loguru import logger

import torchvision
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LoadCache(Dataset):
    def __init__(self, cache_path: str, path: str, word: bool, image_channel: int, resize: list, charset: list,
                 cache_in_memory: bool = False):
        self.cache_path = cache_path
        self.path = path
        self.word = word
        self.ImageChannel = image_channel
        self.resize = resize
        self.charset = charset
        self.charset_map = {c: i for i, c in enumerate(charset)}
        self.cache_in_memory = cache_in_memory
        self._mem_cache = {}
        self.caches = []
        logger.info("\nReading Cache File... ----> {}".format(self.cache_path))

        with open(self.cache_path, 'r', encoding='utf-8') as f:
            self.caches = f.readlines()
        self.caches_num = len(self.caches)
        logger.info("\nRead Cache File End! Caches Num is {}.".format(self.caches_num))

    def __len__(self):
        return self.caches_num

    def __getitem__(self, idx):
        if self.cache_in_memory and idx in self._mem_cache:
            return self._mem_cache[idx]
        try:
            data = self.caches[idx]
            data = data.replace("\r", "").replace("\n", "").split("\t")
            image_name = data[0]
            image_label = data[1]
            image_path = os.path.join(self.path, image_name)
            if not self.word:
                image_label = list(image_label)
            else:
                image_label = [image_label]
            if self.ImageChannel == 1:
                mode = "L"
            else:
                mode = "RGB"
            image = Image.open(image_path).convert(mode)  # shape c, h, w
            image_shape = image.size
            image_height = image_shape[1]
            image_width = image_shape[0]
            width = self.resize[0]
            height = self.resize[1]
            if self.resize[0] == -1:
                if self.word:
                    image = image.resize((height, height))
                else:
                    image = image.resize((int(image_width * (height / image_height)), height))
            else:
                image = image.resize((width, height))
            label = [self.charset_map[item] for item in image_label]
            result = (image, label)
            if self.cache_in_memory:
                self._mem_cache[idx] = result
            return result

        except Exception as e:
            logger.error("\nError: {}, File: {}".format(str(e), self.caches[idx].split("\t")[0]))
            return None, None


class CollateOCR:
    """Top-level collate (picklable on Windows spawn). Do NOT use a bound method."""

    def __init__(self, image_channel: int):
        self.to_tensor = torchvision.transforms.ToTensor()
        if image_channel == 1:
            self.norm_mean = torch.tensor([0.456]).view(1, 1, 1, 1)
            self.norm_std = torch.tensor([0.224]).view(1, 1, 1, 1)
        else:
            self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, batch):
        values = []
        tensors = []
        shapes = []
        max_width = 0
        for img, seq in batch:
            if img is None or seq is None:
                continue
            if len(seq) == 0:
                continue
            t = self.to_tensor(img)
            max_width = max(max_width, t.shape[2])
            values.extend(seq)
            tensors.append(t)
            shapes.append(len(seq))
        if not tensors:
            raise RuntimeError("Empty batch after filtering invalid samples")

        padded = []
        for t in tensors:
            pad_w = max_width - t.shape[2]
            if pad_w > 0:
                t = F.pad(t, (0, pad_w, 0, 0), value=0.0)
            padded.append(t)
        images = torch.stack(padded, dim=0)
        images = (images - self.norm_mean) / self.norm_std
        return [images, torch.as_tensor(values, dtype=torch.float32), torch.as_tensor(shapes, dtype=torch.int32)]


class GetLoader:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.project_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects",
                                         project_name)
        if os.path.exists(self.project_path):
            self.cache_path = os.path.join(self.project_path, "cache")
            if os.path.exists(self.cache_path):
                self.cache_train_path = os.path.join(self.cache_path, "cache.train.tmp")
                self.cache_val_path = os.path.join(self.cache_path, "cache.val.tmp")

                if not os.path.exists(self.cache_train_path):
                    logger.error("\nCache Train File {} is not exists!".format(self.cache_train_path))
                    exit()
                if not os.path.exists(self.cache_val_path):
                    logger.error("\nCache Val File {} is not exists!".format(self.cache_val_path))
                    exit()

            else:
                logger.error("\nCache dir {} is not exists!".format(self.cache_path))
                exit()
        else:
            logger.error("\nProject {} is not exists!".format(project_name))
            exit()

        self.config = Config(project_name)

        self.conf = self.config.load_config()

        self.charset = self.conf['Model']['CharSet']

        logger.info("\nCharsets is {}".format(json.dumps(self.charset, ensure_ascii=False)))

        self.resize = [int(self.conf['Model']['ImageWidth']), int(self.conf['Model']['ImageHeight'])]

        logger.info("\nImage Resize is {}".format(json.dumps(self.resize)))

        self.ImageChannel = self.conf['Model']['ImageChannel']

        self.word = self.conf['Model']['Word']

        self.path = self.conf['System']['Path']

        self.batch_size = self.conf['Train']['BATCH_SIZE']

        self.val_batch_size = self.conf['Train']['TEST_BATCH_SIZE']

        train_conf = self.conf['Train']
        self.num_workers = int(train_conf.get('NUM_WORKERS', 2))
        use_gpu = bool(self.conf['System'].get('GPU', False)) and torch.cuda.is_available()
        self.pin_memory = bool(train_conf.get('PIN_MEMORY', True)) and use_gpu
        self.cache_in_memory = bool(train_conf.get('CACHE_IN_MEMORY', True))

        # Windows spawn + DataLoader：worker>0 时内存缓存在各子进程各自一份，意义不大且更占 RAM
        if self.num_workers > 0 and sys.platform.startswith("win") and self.cache_in_memory:
            logger.warning("\nWindows + NUM_WORKERS>0：已自动关闭 CACHE_IN_MEMORY，避免多进程内存膨胀")
            self.cache_in_memory = False

        if self.ImageChannel not in (1, 3):
            logger.error("ImageChannel must be 1 or 3!")
            exit()

        logger.info("\nImage Path is {}".format(self.path))
        logger.info("\nDataLoader workers={}, pin_memory={}, cache_in_memory={}".format(
            self.num_workers, self.pin_memory, self.cache_in_memory))

        collate_fn = CollateOCR(self.ImageChannel)

        train_dataset = LoadCache(self.cache_train_path, self.path, self.word, self.ImageChannel, self.resize,
                                  self.charset, cache_in_memory=self.cache_in_memory)
        if len(train_dataset) < self.batch_size:
            self.batch_size = len(train_dataset)
        val_dataset = LoadCache(self.cache_val_path, self.path, self.word, self.ImageChannel, self.resize, self.charset,
                                cache_in_memory=self.cache_in_memory)
        if len(val_dataset) < self.batch_size:
            self.val_batch_size = len(val_dataset)

        loader_kwargs = {
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'collate_fn': collate_fn,
            'drop_last': True,
        }
        if self.num_workers > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = 2
            # Windows 下更稳妥
            if sys.platform.startswith("win"):
                loader_kwargs['multiprocessing_context'] = "spawn"

        self.loaders = {
            'train': DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, **loader_kwargs),
            'val': DataLoader(dataset=val_dataset, batch_size=self.val_batch_size, shuffle=True, **loader_kwargs),
        }
