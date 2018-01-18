import pandas as pd
import matplotlib.image as mpimg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import numpy as np

def read_data(path, img_folder_path, is_test=False):
    def load_img(img_path):
        return mpimg.imread(img_path)

    df = pd.read_csv(path)
    df['img_path'] = df['ImageFile'].apply(lambda f: f"{img_folder_path}/{f}")
    df['shape'] = df['img_path'].apply(lambda p: load_img(p).shape)
    df = df[df["shape"].map(lambda s: len(s) == 3)]
    if is_test:
        return df
    else:
        df = df[df["shape"].map(lambda s: s[0] == 256)]
        df = df[df["shape"].map(lambda s: s[1] == 256)]
        df = df[df["shape"].map(lambda s: s[2] == 3)]
        return df

class AestheticsDataset(Dataset):
    def __init__(self, df, is_train):
        self.df = df
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([299, 299]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_transform
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([299, 299]),
                transforms.ToTensor(),
                normalize_transform
            ])

    def load_img(self, img_path):
        return mpimg.imread(img_path)

    def create_targets(self, data_row):
        target_dict = {}
        for k, v in data_row.to_dict().items():
            if k in ['ImageFile', 'img_path']:
                continue
            target_dict[k] = torch.from_numpy(np.array([v]))
        return target_dict

    def get_image(self, image_path):
        data_row = self.df[self.df.img_path == image_path].iloc[0]
        img = self.transform(self.load_img(data_row.img_path))
        img_path = data_row['img_path']
        targets = self.create_targets(data_row)
        return {
            "image": img,
            "image_path": img_path,
            **targets
        }

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        data_row = self.df.iloc[i]
        img = self.transform(self.load_img(data_row.img_path))
        targets = self.create_targets(data_row)
        return {
            "image": img,
            "image_path": data_row.img_path,
            "image_file": data_row.ImageFile,
            **targets
        }


def create_dataloader(df, is_train=True, shuffle=True, batch_size=128):
    dataset = AestheticsDataset(df, is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)