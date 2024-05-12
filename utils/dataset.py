# TODO:Creat dataset
import json
import logging
import os
from glob import glob
import cv2
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BasicDataset(Dataset):
    def __init__(self, n_input, j_dir, img_shape, query, if_transform=True):
        self.n = n_input
        self.dir = j_dir
        self.q = query
        assert os.path.exists(self.dir), "Cannot find {} file".format(self.dir)
        self.data_dict = json.load(open(self.dir, 'r'))
        label_dir = self.data_dict[f'{self.q}_label']
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(label_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        if not if_transform:
            self.trans = transforms.Compose([
                transforms.Resize(img_shape)
            ])
        else:
            self.trans = transforms.Compose([
                transforms.RandomResizedCrop(size=img_shape, scale=(0.75, 1.5)),
            ])

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, image):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        return image

    def __getitem__(self, i):
        data_dict = self.data_dict
        idx = self.ids[i]
        output_dict = dict()
        seed = np.random.randint(2147483647)

        for j in range(1, 1+self.n):
            imgs_dir = data_dict[f'{j}_{self.q}_img']
            img_file = glob(imgs_dir + idx + "*")
            img_name = img_file[0].split('.')[-1]
            if img_name == 'tif':
                img = tiff.imread(img_file[0])
            else:
                img = cv2.imdecode(np.fromfile(img_file[0], dtype=np.uint8), -1)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.preprocess(img)
            torch.manual_seed(seed)
            img = self.trans(img)
            output_dict[f'image{j}'] = img / 255  # normalized

        mask_file = glob(data_dict[f'{self.q}_label'] + idx + "*")
        mask = cv2.imdecode(np.fromfile(mask_file[0], dtype=np.uint8), -1)
        mask = self.preprocess(mask)
        torch.manual_seed(seed)
        mask = self.trans(mask)
        output_dict['mask'] = mask

        return output_dict
