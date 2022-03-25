
import numpy as np
import os
import torch
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AVDRIVEloader(torch.utils.data.Dataset):
    def __init__(self, dir_img=None, transform=None):
        super(AVDRIVEloader, self).__init__()

        self.dir_img = dir_img
        self.transform = transform

        fn_img = os.listdir(dir_img)
        fn_img.sort()

        idx = [*range(0, len(fn_img))]
        self.fn_img = [fn_img[i] for i in idx]

        self.nums_img = len(self.fn_img)

    def __len__(self):
        return len(self.fn_img)

    def __getitem__(self, idx):
        img_name = self.fn_img[idx]
        img = Image.open(os.path.join(self.dir_img, img_name))
        img = self.transform(img)

        return img


