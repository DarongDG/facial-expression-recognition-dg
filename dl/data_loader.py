import os

import numpy as np
import pandas as pd
import torch as t
import ipdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils

# load dataa
from dl.custom_fer_loaders import DataFER

df = pd.read_csv("fer2013.csv")
# split train test valid
df_train = df[df.Usage == 'Training'].drop(['Usage'], axis=1).reset_index().drop(columns="index")
df_valid = df[df.Usage == 'PrivateTest'].drop(['Usage'], axis=1).reset_index().drop(columns="index")
df_test = df[df.Usage == 'PublicTest'].drop(['Usage'], axis=1).reset_index().drop(columns="index")
train_images = df_train.iloc[:, 1]
train_labels = df_train.iloc[:, 0]
valid_images = df_train.iloc[:, 1]
valid_labels = df_train.iloc[:, 0]
test_images = df_test.iloc[:, 1]
test_labels = df_test.iloc[:, 0]


train_trans = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(48, padding=4, padding_mode='reflect'),
        # maybe useful
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize
        transforms.Normalize((0.5), (0.5), inplace=True)
    ])

# transform data for validation (just greyscale and normalize)
val_trans = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])


class CustomDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.data_location = params.data_location
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size
        # self.data_location = params.data_location

    def train_dataloader(self):
        # creates a DeepCoastalDataset object
        dataset = DataFER(train_images, train_labels, train_trans)
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=1,
        )

    def val_dataloader(self):
        # creates a DeepCoastalDataset object
        dataset = DataFER(valid_images, valid_labels, val_trans)
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=1,
        )

    def test_dataloader(self):

        dataset = DataFER(train_images, train_labels, train_trans)
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=1,
        )


def test():

    data_module = CustomDataModule()
    train_dataloader = data_module.train_dataloader()
    for i, (x, y) in enumerate(train_dataloader):
        print(i, x.shape, y.shape)
    """
    train_dl, test_dl = get_loaders(
        "/mnt/tmp/multi_channel_train_test",
        32,
        64,
        t.device("cuda" if t.cuda.is_available() else "cpu"),
        in_seq_len=8,
        out_seq_len=4,
    )
    for i, (x, y) in enumerate(tqdm(train_dl)):
        # plt.imshow(x[0, 0, 0].cpu())
        # plt.show()
        # print(x.shape)
        # return
        # print(f"iteration: {i}")
        pass
    """
    # reads file in h5 format

if __name__ == "__main__":
    test()

