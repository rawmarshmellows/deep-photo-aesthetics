from tqdm import trange
from collections import defaultdict
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch, torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from collections import OrderedDict
from tqdm import tqdm
from pathlib import Path
from scipy.stats.stats import pearsonr

## Create dataset

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

import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self, resnet, n_features=12):
        super().__init__()
        self.initial_layers = nn.Sequential(*list(resnet.children())[:4])
        self.bottlenecks = []
        self.attribute_weights = nn.Linear(15104, n_features)

        # Extract the bottleneck layers
        for i, mod in enumerate(list(resnet.children())):
            if isinstance(mod, nn.Sequential):
                for bn in mod:
                    self.bottlenecks.append(bn)

        # Set the resnet weights to not update
        for param in resnet.parameters():
            param.requires_grad = False

    def forward(self, inp):
        all_feature_maps = []
        output = self.initial_layers(inp)

        # Loop to extract the outputs of the bottleneck layers from resnet
        for bn in self.bottlenecks:
            output = bn(output)
            kernel_size = (output.size()[2], output.size()[3])
            feature_maps = F.avg_pool2d(output, kernel_size)
            all_feature_maps.append(feature_maps)

        # Global pool
        features = torch.cat(all_feature_maps, dim=1).squeeze()
        if len(features.size()) == 1:
            features = features.unsqueeze(0)

        # Use features to predict scores
        attribute_scores = self.attribute_weights(features)
        attr = F.tanh(attribute_scores[:, :9])
        non_neg_attr = F.sigmoid(attribute_scores[:, 9:])
        predictions = torch.cat([attr, non_neg_attr], dim=1)
        return predictions

class MyNet2(nn.Module):
    def __init__(self, resnet, n_features=12):
        super().__init__()
        self.model = resnet
        for param in self.model.parameters():
            param.requires_grad = False
        self.attribute_weights = nn.Linear(4 * 512, n_features)
        self.model.fc = self.attribute_weights
        self.model.avgpool = nn.AvgPool2d(kernel_size=10)

    def forward(self, inp):
        output = self.model(inp)
        return output

class PoolFeatures(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        kernel_size = (inp.size()[2], inp.size()[3])
        self.feature_maps = F.avg_pool2d(inp, kernel_size)
        return inp

class FeaturesMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        self.feature_maps = inp
        return inp

class MyNet3(nn.Module):
    def __init__(self, resnet, n_features=12):
        super().__init__()
        self.model = nn.Sequential(*list(resnet.children())[:4])
        self.all_features = []
        self.all_pooled_features = []
        self.attribute_weights = nn.Linear(15104, n_features)

        count = 0
        for i, mod in enumerate(list(resnet.children())):
            # Extract the bottleneck layers
            if isinstance(mod, nn.Sequential):
                for bn in mod:
                    self.model.add_module(f"bn_{count}", bn)

                    # Use a "Transparent layer"
                    pooled_feature_map = PoolFeatures()
                    feature_map = FeaturesMap()
                    self.model.add_module(f"pooled_feature_{count}", pooled_feature_map)
                    self.model.add_module(f"feature_map_{count}", feature_map)
                    self.all_pooled_features.append(pooled_feature_map)
                    self.all_features.append(feature_map)
                    count += 1

    def forward(self, inp):
        _ = self.model(inp)
        features = torch.cat([pool_fp.feature_maps for pool_fp in self.all_pooled_features], dim=1).squeeze()
        features = F.dropout(features, p=0.2)
        if len(features.size()) == 1:
            features = features.unsqueeze(0)

        # Use features to predict scores
        attribute_scores = self.attribute_weights(features)
        attr = F.tanh(attribute_scores[:, :9])
        non_neg_attr = F.sigmoid(attribute_scores[:, 9:])
        predictions = torch.cat([attr, non_neg_attr], dim=1)
        return predictions

def create_all_targets(data):
    targets = []
    for k in attr_keys:
        targets.append(data[k])
    for k in non_neg_attr_keys:
        targets.append(data[k])
    targets = Variable(torch.cat(targets, dim=1)).float()
    return targets

if "__main__" == __name__:
    use_cuda = torch.cuda.is_available()
    train = read_data("/home/kevin/workspace/aesthetic-attributes-maps/imgListTrainRegression_.csv",
                      "/home/kevin/workspace/aesthetic-attributes-maps/datasetImages/images")

    val = read_data("/home/kevin/workspace/aesthetic-attributes-maps/imgListValidationRegression_.csv",
                      "/home/kevin/workspace/aesthetic-attributes-maps/datasetImages/images")

    test = read_data("/home/kevin/workspace/aesthetic-attributes-maps/imgListTestNewRegression_.csv",
                     "/home/kevin/workspace/aesthetic-attributes-maps/datasetImages/images",
                     is_test=True)
    batch_size = 16
    train_dataset = create_dataloader(train, batch_size=batch_size, is_train=True, shuffle=True)
    val_dataset = create_dataloader(val, batch_size=batch_size, is_train=False, shuffle=False)

    resnet = models.resnet50(pretrained=True)
    net = MyNet3(resnet, n_features=12)
    if use_cuda:
        resnet = resnet.cuda()
        net = net.cuda()

    ## Training loop
    attr_keys = ['BalancingElements', 'ColorHarmony', 'Content', 'DoF',
                 'Light', 'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor']
    non_neg_attr_keys = ['Repetition', 'Symmetry', 'score']
    all_keys = attr_keys + non_neg_attr_keys

    loss_weights = {
            "Content": 0.5,
            "Object": 0.5,
            "Symmetry": 0.0,
            "RuleOfThirds": 0.5,
            "Light": 0.5,
            "MotionBlur": 0.0,
            "DoF": 0.5,
            "ColorHarmony": 0.5,
            "BalacingElements": 0.5,
            "VividColor": 0.5,
            "score": 2,
            "Repetition": 0.0
        }

    # loss_weights = {
    #         "Content": 0.0,
    #         "Object": 0.0,
    #         "Symmetry": 0.0,
    #         "RuleOfThirds": 0.0,
    #         "Light": 0.0,
    #         "MotionBlur": 0.0,
    #         "DoF": 0.0,
    #         "ColorHarmony": 0.0,
    #         "BalacingElements": 0.0,
    #         "VividColor": 1.0,
    #         "score": 0.0,
    #         "Repetition": 0.0
    #     }

    ignored_params = list(map(id, net.attribute_weights.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         net.parameters())

    optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': net.attribute_weights.parameters(), 'lr': 1e-5, 'weight_decay': 1e-5}
            ], lr=1e-6)

    # optimizer = torch.optim.Adam(params=net.attribute_weights.parameters(), lr=1e-5)
    # optimizer = torch.optim.Adagrad(params=net.attribute_weights.parameters(), lr=1e-4)
    # optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-5)
    criterion = nn.MSELoss(reduce=False)


    weights = torch.zeros(1, len(attr_keys + non_neg_attr_keys))
    for i, attr in enumerate(attr_keys + non_neg_attr_keys):
        weight = loss_weights[attr]
        weights[0, i] = weight

    if use_cuda:
        weights = weights.cuda()

    train_loss = []
    train_corr = []
    val_loss = []
    val_corr = []
    save_path = Path("pytorch_model/resnet-GAP-features-FT3")
    save_path.mkdir(parents=True, exist_ok=True)
    for epoch in tqdm(range(15)):
        train_loss_data_for_df = defaultdict(list)
        train_correlation_data_for_df = defaultdict(list)
        val_loss_data_for_df = defaultdict(list)
        val_correlation_data_for_df = defaultdict(list)

        train_data_tqdm = tqdm(train_dataset)
        for data in train_data_tqdm:
            net.train()
            inp = Variable(data['image'])
            if use_cuda:
                inp = inp.cuda()
            predictions = net(inp)
            if use_cuda:
                targets = create_all_targets(data).cuda()
            else:
                targets = create_all_targets(data)

            loss = criterion(predictions, targets)
            total_loss_per_sample = torch.sum(loss.data * weights, dim=1)
            masked_loss = loss.data * weights
            loss_by_attribute = torch.sum(loss, dim=0).unsqueeze(0)
            for i in range(loss.size()[0]):
                for j, k in enumerate(attr_keys + non_neg_attr_keys):
                    train_loss_data_for_df[k].append(masked_loss[i, j])
                train_loss_data_for_df["total_loss"].append(total_loss_per_sample[i])
                train_loss_data_for_df["epoch"].append(epoch)

            for j, k in enumerate(all_keys):
                train_correlation_data_for_df[k].append(pearsonr(predictions[:, j].data.cpu().numpy(),
                                                                 targets[:, j].data.cpu().numpy())[0])
            train_correlation_data_for_df["epoch"].append(epoch)
            optimizer.zero_grad()

            # Method 1:
            torch.autograd.backward(loss_by_attribute, weights)

            # Method 2:
            # masked_loss = torch.sum(masked_loss)
            # masked_loss.backward()

            optimizer.step()

        train_correlation_df = pd.DataFrame(train_correlation_data_for_df)
        train_corr.append(train_correlation_df)
        train_loss_df = pd.DataFrame(train_loss_data_for_df)
        train_loss.append(train_loss_df)

        print(f"Training Loss Breakdown:\n{train_loss_df.mean()}")
        print(f"Training Correlation Breakdown:\n{train_correlation_df.mean()}")
        for data in tqdm(val_dataset):
            net.eval()
            inp = Variable(data['image'])
            if use_cuda:
                inp = inp.cuda()
            predictions = net(inp)
            if use_cuda:
                targets = create_all_targets(data).cuda()
            else:
                targets = create_all_targets(data)

            loss = criterion(predictions, targets)
            total_loss_per_sample = torch.sum(loss.data * weights, dim=1)
            masked_loss = loss.data * weights
            val_loss_data_for_df["image_file"].extend(data['image_file'])
            for i in range(loss.size()[0]):
                for j, k in enumerate(attr_keys + non_neg_attr_keys):
                    val_loss_data_for_df[k].append(masked_loss[i, j])
                val_loss_data_for_df["total_loss"].append(total_loss_per_sample[i])
                val_loss_data_for_df["epoch"].append(epoch)

            for j, k in enumerate(all_keys):
                val_correlation_data_for_df[k].append(pearsonr(predictions[:, j].data.cpu().numpy(),
                                                                 targets[:, j].data.cpu().numpy())[0])
            val_correlation_data_for_df["epoch"].append(epoch)
        val_correlation_df = pd.DataFrame(val_correlation_data_for_df)
        val_corr.append(val_correlation_df)
        val_loss_df = pd.DataFrame(val_loss_data_for_df)
        val_loss.append(val_loss_df)
        print()
        print(f"Validation Loss Breakdown:\n{val_loss_df.mean()}")
        print(f"Validation Correlation Breakdown:\n{val_correlation_df.mean()}")

        torch.save(net.state_dict(), f"{save_path}/epoch_{epoch}.loss_{val_loss_df.mean()['total_loss']}.pth")

    # In[22]:val
    train_loss = pd.DataFrame(pd.concat(train_loss))
    val_loss = pd.DataFrame(pd.concat(val_loss))
    train_loss.to_csv(f"{save_path}/train_results.csv")
    val_loss.to_csv(f"{save_path}/val_results.csv")

    train_corr = pd.DataFrame(pd.concat(train_corr))
    val_corr = pd.DataFrame(pd.concat(val_corr))
    train_corr.to_csv(f"{save_path}/train_corr_results.csv")
    val_corr.to_csv(f"{save_path}/val_corr_results.csv")


    # In[21]:


    # In[ ]:




