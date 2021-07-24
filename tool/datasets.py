import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_ = [transforms.Normalize([0.5], [0.5], [0.5])]

class NpyDataset(Dataset):
    def __init__(self, root="data", transforms_=transform_, mode="train"):
        self.transform = transforms.Compose(transforms_)
        # prepare accessible dataset
        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*/*.*'))
        self.labels = [file.split('\\')[2] for file in self.files]
        self.dict = {'000': 0, '001': 1, '002': 2, '003': 3, '004': 4}

    def __getitem__(self, index):
        sample = np.load(self.files[index])
        sample[:, 1, :, :, :] = 0
        frames = 48     # 抽取的帧数
        sample = extract_frame(sample, frames)
        sample = np.resize(sample, (1,3,frames,34))
        sample = torch.Tensor(np.squeeze(sample))

        # get the transformed data and label
        if self.transform is not None:
            sample = self.transform(sample)
            sample = sample.to(device)
        label = self.dict[self.labels[index]]
        label = torch.tensor(label).to(device)

        return sample, label

    def __len__(self):
        return len(self.files)


# 帧数抽取工具，按一定帧数间隔抽取数据
def extract_frame(data, num):
    N, C, T, V, M = data.shape
    sample = np.zeros((N, C, num, V, M))
    for i in range(num):
        sample[:, :, i, :, :] = data[:, :, int(T*i/num), :, :]

    return sample

