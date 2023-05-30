# encoding: utf-8
"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os


N_CLASSES = 7
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


class CheXpertDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            root_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(CheXpertDataset, self).__init__()
        file = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.images = file['image'].values
        self.labels = file.iloc[:, 1:].values.astype(int)
        self.transform = transform

        print('Total # images:{}, labels:{}'.format(len(self.images), len(self.labels)))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        items = self.images[index]  # image_file name
        image_name = os.path.join(self.root_dir, self.images[index] + ".jpg")
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        # train log
        # print(items, index, image, label)

        if self.transform is not None:
            image = self.transform(image)
        return items, index, image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.images)


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

