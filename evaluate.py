import logging
import os
import sys
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset import NextDayFireDataset
import matplotlib.pyplot as plt
import numpy as np
import random
import gc
from models.unet import UNet
from models.attention_unet import AttentionUNet, ResidualAttentionUNet
import wandb
from train import (
    load_checkpoint,
    seed_all,
    evaluate,
    model_mapping,
)
from models.swin_unet.vision_transformer import FireSwinUnet
from models.swin_unet.swin_config import get_config
from models.swin_attention_unet.attention_swin_unet import SwinAttentionUnet
from models.focalnet.focalnet import FireFocalNet
from models.revcol.revcol import revcol_tiny
from configs.swin_attention_unet import (
    get_swin_unet_attention_configs,
    get_ca_swin_unet_attention_configs,
    get_ca_focal_swin_unet_attention_configs,
    get_focal_swin_unet_attention_12_configs,
)

torch.set_printoptions(threshold=100000000, edgeitems=10)
# set seed
seed_all(42)


# Load model
# Unet
# load_model = 'checkpoints/unet/checkpoint_epoch7_posweight3.pth'
# Attention unet
# load_model = 'checkpoints/att_unet/checkpoint_epoch12_0.267_att_unet.pth'
# Attention Swin unet
# load_model = 'checkpoints/att_swin_unet/checkpoint_epoch24_0.pth'
# Focal unet
# load_model = 'checkpoints/focal_unet/checkpoint_epoch_26_focalunet.pth'
# load_model = 'checkpoints/ca_att_swin_focal/checkpoint_epoch_10_0.pth'
# Swin_Unet
load_model = 'checkpoints/swin_6features/checkpoint_epoch35_0.pth'


model_type = load_model.split('/')[1]
if model_type == 'unet':
    state_dict = torch.load(load_model, map_location=torch.device('cpu'))
    input_features = ['NDVI', 'elevation', 'PrevFireMask']
    del state_dict['mask_values']
else:
    state_dict, _, _, _, input_features = load_checkpoint(load_model)

# Set parameters
if input_features:
    n_channels = len(input_features)
else:
    n_channels = 4
    input_features = ['sph', 'NDVI', 'elevation', 'PrevFireMask']
n_classes = 1
batch_size = 32  # This should not be too small otherwise the class weights might cause index error
amp = False  # This needs to be false for cpu / mps
sampling_method = 'original'
sampling_mode = 'test'
use_augmented = True
save_predictions = False

# Datasets
if use_augmented:
    train_data_file_pattern = (
        '../dataset/next-day-fire-2012-2023/northamerica_2012-2023/train/*.tfrecord'
    )
    val_data_file_pattern = (
        '../dataset/next-day-fire-2012-2023/northamerica_2012-2023/val/*.tfrecord'
    )
    test_data_file_pattern = (
        '../dataset/next-day-fire-2012-2023/northamerica_2012-2023/test/*.tfrecord'
    )
    print('Using augmented dataset')
else:
    train_data_file_pattern = (
        '../dataset/next-day-fire/next_day_wildfire_spread_train_*.tfrecord'
    )
    val_data_file_pattern = (
        '../dataset/next-day-fire/next_day_wildfire_spread_eval_*.tfrecord'
    )
    test_data_file_pattern = (
        '../dataset/next-day-fire/next_day_wildfire_spread_test_*.tfrecord'
    )
    print('Using original dataset')
train_data_file_names = tf.io.gfile.glob(train_data_file_pattern)
val_data_file_names = tf.io.gfile.glob(val_data_file_pattern)
test_data_file_names = tf.io.gfile.glob(test_data_file_pattern)
# Make tf datasets
train_data = tf.data.TFRecordDataset(train_data_file_names)
val_data = tf.data.TFRecordDataset(val_data_file_names)
test_data = tf.data.TFRecordDataset(test_data_file_names)

train_set = NextDayFireDataset(
    train_data,
    limit_features_list=input_features,
    clip_normalize=True,
    sampling_method=sampling_method,
    mode=sampling_mode,
)
val_set = NextDayFireDataset(
    val_data,
    limit_features_list=input_features,
    clip_normalize=True,
    sampling_method=sampling_method,
    mode=sampling_mode,
)
test_set = NextDayFireDataset(
    test_data,
    limit_features_list=input_features,
    clip_normalize=True,
    sampling_method=sampling_method,
    mode=sampling_mode,
)
# n_train = len(train_set)
# n_val = len(val_set)

loader_args = dict(
    batch_size=batch_size,
)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Test metrics and loss
train_loader = DataLoader(train_set, shuffle=False, drop_last=False, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)

if model_type in ['swin_unet', 'swin_6features']:
    config = get_config(args=None)
    model = FireSwinUnet(config=config, img_size=64, num_classes=n_classes)
elif model_type == 'att_swin_unet':
    config = get_swin_unet_attention_configs()
    model = SwinAttentionUnet(config=config, num_classes=n_classes)
elif model_type == 'ca_att_swin_unet':
    config = get_ca_swin_unet_attention_configs()
    model = SwinAttentionUnet(config=config, num_classes=n_classes)
elif model_type in ['focal_att_swin_unet', 'ca_att_swin_focal']:
    config = get_ca_focal_swin_unet_attention_configs()
    model = SwinAttentionUnet(config=config, num_classes=n_classes)
elif model_type == 'focal_att_swin_unet_12':
    config = get_focal_swin_unet_attention_12_configs()
    model = SwinAttentionUnet(config=config, num_classes=n_classes)
elif model_type == 'focal_unet':
    model = FireFocalNet(
        in_chans=12,
        depths=[2, 2, 2, 2],
        embed_dim=96,
        use_layerscale=True,
        use_postln=True,
    )
elif model_type == 'revcol':
    model = revcol_tiny(
        save_memory=True, inter_supv=True, drop_path=0.1, num_classes=1, kernel_size=3
    )
else:
    model = model_mapping[model_type](
        n_channels,
        n_classes,
    )

model.load_state_dict(state_dict)
model = model.to(memory_format=torch.channels_last)

dice_score, auc_score, val_loss, precision, recall, f1 = evaluate(
    model,
    val_loader,
    device,
    amp,
    batch_size,
    n_val=len(val_set),
    save_predictions=save_predictions,
)

print(f'Result model: {load_model}')
print(f'Validation Dice score: {dice_score}')
print(f'Validation AUC score: {auc_score}')
print(f'Validation Loss: {val_loss}')
print(f'Validation Precision: {precision}')
print(f'Validation Recall: {recall}')
print(f'Validation F1 score: {f1}')

gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test metrics and loss
val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)

if model_type == 'swin_unet':
    config = get_config(args=None)
    model = FireSwinUnet(config=config, img_size=64, num_classes=n_classes)
elif model_type == 'att_swin_unet':
    config = get_swin_unet_attention_configs()
    model = SwinAttentionUnet(config=config, num_classes=n_classes)
elif model_type == 'ca_att_swin_unet':
    config = get_ca_swin_unet_attention_configs()
    model = SwinAttentionUnet(config=config, num_classes=n_classes)
elif model_type == 'focal_att_swin_unet':
    config = get_ca_focal_swin_unet_attention_configs()
    model = SwinAttentionUnet(config=config, num_classes=n_classes)
elif model_type == 'focal_att_swin_unet_12':
    config = get_focal_swin_unet_attention_12_configs()
    model = SwinAttentionUnet(config=config, num_classes=n_classes)
elif model_type == 'focal_unet':
    model = FireFocalNet(
        in_chans=12,
        depths=[2, 2, 2, 2],
        embed_dim=96,
        use_layerscale=True,
        use_postln=True,
    )
elif model_type == 'revcol':
    model = revcol_tiny(
        save_memory=True, inter_supv=True, drop_path=0.1, num_classes=1, kernel_size=3
    )
else:
    model = model_mapping[model_type](
        n_channels,
        n_classes,
    )

model.load_state_dict(state_dict)
model = model.to(memory_format=torch.channels_last)


# loaders = [
#     # train_loader,
#     val_loader,
#     test_loader,
# ]
# sets = [
#     # train_set,
#     val_set,
#     test_set,
# ]
# total_batch_len = 0
# for loader, dataset in zip(loaders, sets):
#     data_len = len(dataset)
#     dice_scores = []
#     auc_scores = []
#     val_losses = []
#     precisions = []
#     recalls = []
#     f1_scores = []
#     dice_score, auc_score, val_loss, precision, recall, f1 = evaluate(
#         model,
#         loader,
#         device,
#         amp,
#         batch_size,
#         n_val=data_len,
#         save_predictions=save_predictions,
#     )
#     dice_scores.append(dice_score * data_len)
#     auc_scores.append(auc_score * data_len)
#     val_losses.append(val_loss * data_len)
#     precisions.append(precision * data_len)
#     recalls.append(recall * data_len)
#     f1_scores.append(f1 * data_len)
#     total_batch_len += len(data_len)

#     # calculate average per batch
#     dice_score = sum(dice_scores) / total_batch_len
#     auc_score = sum(auc_scores) / total_batch_len
#     val_loss = sum(val_losses) / total_batch_len
#     precision = sum(precisions) / total_batch_len
#     recall = sum(recalls) / total_batch_len
#     f1 = sum(f1_scores) / total_batch_len

dice_score, auc_score, val_loss, precision, recall, f1 = evaluate(
    model,
    val_loader,
    device,
    amp,
    batch_size,
    n_val=len(val_set),
    save_predictions=save_predictions,
)

print(f'Result model: {load_model}')
print(f'Validation Dice score: {dice_score}')
print(f'Validation AUC score: {auc_score}')
print(f'Validation Loss: {val_loss}')
print(f'Validation Precision: {precision}')
print(f'Validation Recall: {recall}')
print(f'Validation F1 score: {f1}')

gc.collect()
