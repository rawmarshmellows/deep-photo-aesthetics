import json
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
from pathlib import Path
import argparse
from collections import defaultdict
from utils.data import read_data, create_dataloader
from model.resnet_FT import resnet_gap_features
from utils.cuda import cudarize
import torch
import torch.nn as nn
from torch.autograd import Variable


def setup_data(train_path, val_path, img_folder_path, batch_size):
    train = read_data(train_path, img_folder_path)
    val = read_data(val_path, img_folder_path)
    train_dataset = create_dataloader(train, batch_size=batch_size, is_train=True, shuffle=True)
    val_dataset = create_dataloader(val, batch_size=batch_size, is_train=False, shuffle=False)
    return train_dataset, val_dataset


def setup_model(use_cuda):
    model = resnet_gap_features()
    model = cudarize(model, use_cuda)
    return model


def create_all_targets(data, attributes):
    targets = []
    for attr in attributes:
        targets.append(data[attr])
    targets = Variable(torch.cat(targets, dim=1)).float()
    return targets


def update_results(epoch, predictions, targets, loss, weights, all_attributes, loss_data_for_df, corr_data_for_df):
    total_loss_per_sample = torch.sum(loss.data * weights, dim=1)
    masked_loss = loss.data * weights
    current_batch_size = loss.size()[0]
    for i in range(current_batch_size):
        for j, k in enumerate(all_attributes):
            loss_data_for_df[k].append(masked_loss[i, j])
        loss_data_for_df["total_loss"].append(total_loss_per_sample[i])
        loss_data_for_df["epoch"].append(epoch)

    for j, k in enumerate(all_attributes):
        corr_data_for_df[k].append(pearsonr(predictions[:, j].data.cpu().numpy(), targets[:, j].data.cpu().numpy())[0])
    corr_data_for_df["epoch"].append(epoch)


def train(train, val, model, loss_weights, n_epochs, use_cuda, save_path,
          fc_lr, fine_tune_lr):
    save_path = Path(save_path)
    attribute_keys = ['BalancingElements', 'ColorHarmony', 'Content', 'DoF',
                      'Light', 'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor']
    non_negative_attribute_keys = ['Repetition', 'Symmetry', 'score']
    all_attributes = attribute_keys + non_negative_attribute_keys
    ignored_params = list(map(id, model.attribute_weights.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())

    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.attribute_weights.parameters(), 'lr': fc_lr, 'weight_decay': 1e-5}
    ], lr=fine_tune_lr)
    criterion = nn.MSELoss(reduce=False)

    # Define the weights that are needed on later
    weights = torch.zeros(1, len(all_attributes))
    for i, attr in enumerate(all_attributes):
        weight = loss_weights[attr]
        weights[0, i] = weight

    weights = cudarize(weights, use_cuda)

    train_loss = []
    train_corr = []
    val_loss = []
    val_corr = []
    for epoch in tqdm(range(n_epochs)):
        train_loss_data_for_epoch = defaultdict(list)
        train_correlation_data_for_epoch = defaultdict(list)
        val_loss_data_for_epoch = defaultdict(list)
        val_correlation_data_for_epoch = defaultdict(list)

        for data in tqdm(train):
            model.train()
            inp = cudarize(Variable(data['image']), use_cuda)
            predictions = model(inp)
            targets = cudarize(create_all_targets(data, all_attributes), use_cuda)
            loss = criterion(predictions, targets)
            loss_by_attribute = torch.sum(loss, dim=0).unsqueeze(0)

            # Update results
            update_results(epoch, predictions, targets, loss, weights,
                           all_attributes, train_loss_data_for_epoch,
                           train_correlation_data_for_epoch)

            # Update gradients
            optimizer.zero_grad()
            
            # The two methods below are equivalent!
            
            # Method 1:
            torch.autograd.backward(loss_by_attribute, weights)

            # Method 2:
            # masked_loss = loss_by_attribute * weights
            # masked_loss = torch.sum(masked_loss)
            # masked_loss.backward()

            optimizer.step()

        train_correlation_df_for_epoch = pd.DataFrame(train_correlation_data_for_epoch)
        train_corr.append(train_correlation_df_for_epoch)
        train_loss_df_for_epoch = pd.DataFrame(train_loss_data_for_epoch)
        train_loss.append(train_loss_df_for_epoch)

        print(f"\nTraining Loss Breakdown:\n{train_loss_df_for_epoch.mean()}")
        print(f"\nTraining Correlation Breakdown:\n{train_correlation_df_for_epoch.mean()}")

        for data in tqdm(val):
            model.eval()
            inp = cudarize(Variable(data['image']), use_cuda)
            predictions = model(inp)
            targets = cudarize(create_all_targets(data, all_attributes), use_cuda)
            loss = criterion(predictions, targets)

            # Update results
            update_results(epoch, predictions, targets, loss, weights,
                           all_attributes, val_loss_data_for_epoch,
                           val_correlation_data_for_epoch)

        val_correlation_df_for_epoch = pd.DataFrame(val_correlation_data_for_epoch)
        val_corr.append(val_correlation_df_for_epoch)
        val_loss_df_for_epoch = pd.DataFrame(val_loss_data_for_epoch)
        val_loss.append(val_loss_df_for_epoch)
        print(f"\nValidation Loss Breakdown:\n{val_loss_df_for_epoch.mean()}")
        print(f"\nValidation Correlation Breakdown:\n{val_correlation_df_for_epoch.mean()}")


        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f"{save_path}/epoch_{epoch}.loss_{val_loss_df_for_epoch.mean()['total_loss']}.pth")


    train_loss = pd.DataFrame(pd.concat(train_loss))
    val_loss = pd.DataFrame(pd.concat(val_loss))
    train_loss.to_csv(f"{save_path}/train_results.csv")
    val_loss.to_csv(f"{save_path}/val_results.csv")

    train_corr = pd.DataFrame(pd.concat(train_corr))
    val_corr = pd.DataFrame(pd.concat(val_corr))
    train_corr.to_csv(f"{save_path}/train_corr_results.csv")
    val_corr.to_csv(f"{save_path}/val_corr_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", default="config.json")
    opts = parser.parse_args()
    with open(opts.config_file_path, "r") as fp:
        config = json.load(fp)
    train_dataset, val_dataset = setup_data(config['train_path'],
                                            config['val_path'],
                                            config['img_folder_path'],
                                            config['batch_size'],
                                            )
    model = setup_model(config['use_cuda'])
    train(train_dataset, val_dataset, model,
          config['loss_weights'], config['n_epochs'],
          config['use_cuda'], config['save_path'],
          config['fc_lr'], config['fine_tune_lr'])

