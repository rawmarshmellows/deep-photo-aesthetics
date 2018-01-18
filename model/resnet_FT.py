import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models


class PoolFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_maps = None

    def forward(self, inp):
        kernel_size = (inp.size()[2], inp.size()[3])
        self.feature_maps = F.avg_pool2d(inp, kernel_size)
        return inp


class FeaturesMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_maps = None

    def forward(self, inp):
        self.feature_maps = inp
        return inp


class ResNetGAPFeatures(nn.Module):
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

                    # Use "Transparent layers and save references to their objects for later use"
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

        # The first 9 scores reflect:
        # 'BalancingElements', 'ColorHarmony', 'Content', 'DoF','Light',
        #  'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor'
        # which are between values -1 and 1, hence the tanh non-linearity
        attr = F.tanh(attribute_scores[:, :9])

        # The last 3 scores reflect
        # 'Repetition', 'Symmetry', 'score' which are between values 0 and 1
        # hence the sigmoid non-linearity
        non_neg_attr = F.sigmoid(attribute_scores[:, 9:])
        predictions = torch.cat([attr, non_neg_attr], dim=1)
        return predictions

def resnet_gap_features():
    resnet = models.resnet50(pretrained=True)
    model = ResNetGAPFeatures(resnet)
    return model
