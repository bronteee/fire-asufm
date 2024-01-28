# Bronte Sihan Li, 2024

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
from dataset import NextDayFireDataset
import matplotlib.pyplot as plt
import numpy as np
import random
import gc
import pytorch_warmup as warmup
from torch.utils.checkpoint import checkpoint

import wandb
from wandb import Artifact

sys.path.append(os.path.dirname(__file__))
from metrics_loss import (
    dice_coeff,
    get_auc,
    get_precision,
    get_recall,
    get_weighted_f1,
    FocalTverskyLoss,
)

POS_THRESHOLD = 0.5
REVCOL_THRESHOLD = 0.01

dir_checkpoint = Path('checkpoints')
log_dir = Path('logs')


def seed_all(seed):
    # Pytorch Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set global random seed
    tf.random.set_seed(seed)
    # Set operation-level random seeds
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def train_next_day_fire(
    train_data: tf.data.Dataset,
    val_data: tf.data.Dataset,
    model,
    device,
    dir_checkpoint: Path = Path('checkpoints'),
    starting_epoch: int = 1,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
    optimizer: str = 'adamw',
    optimizer_state_dict=None,
    weight_decay: float = 0.05,
    momentum: float = 0.9,
    gradient_clipping: float = 1.0,
    pos_weight: float = 3.0,
    limit_features: list = ['sph', 'NDVI', 'elevation', 'PrevFireMask'],
    loss_function: str = 'bce',
    activation: str = 'relu',
    sampling_method: str = 'random_crop',
    skip_eval: bool = False,
    use_warmup: bool = True,
    warmup_lr_init: float = 5e-7,
    use_checkpointing: bool = False,
):
    # Create dataset
    train_set = NextDayFireDataset(
        train_data,
        limit_features_list=limit_features,
        clip_normalize=True,
        sampling_method=sampling_method,
    )
    val_set = NextDayFireDataset(
        val_data,
        limit_features_list=limit_features,
        clip_normalize=True,
        sampling_method=sampling_method,
    )
    n_train = len(train_set)
    n_val = len(val_set)

    # Create data loaders
    val_batch_size = 64
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size,
    )
    if skip_eval is not True:
        val_loader = DataLoader(
            val_set, shuffle=False, drop_last=True, batch_size=val_batch_size
        )
    else:
        val_loader = None
    # Initialize logging
    experiment = wandb.init(
        project=model.__class__.__name__,
        resume='allow',
        anonymous='must',
        entity='fire-dream',
    )
    experiment.config.update(
        dict(
            starting_epoch=starting_epoch,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_percent=val_percent,
            save_checkpoint=save_checkpoint,
            img_scale=img_scale,
            amp=amp,
            input_features=limit_features,
            model=model.__class__.__name__,
            optimizer=optimizer,
            loss_function=loss_function,
            activation=activation,
        )
    )
    os.makedirs(log_dir, exist_ok=True)

    logging.info(
        f'''Starting training:
        Starting epoch:  {starting_epoch}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Positive weight: {pos_weight}
        Input features:  {limit_features}
    '''
    )

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            amsgrad=True,
            foreach=True,
        )
    elif optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            amsgrad=True,
            foreach=True,
        )
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            foreach=True,
        )
    elif optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            foreach=True,
        )
    else:
        raise NotImplementedError
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)
    num_steps = len(train_loader) * epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps
    )
    global_step = 0
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    # Training
    for epoch in range(1, epochs + 1):
        (
            val_dice_score,
            val_auc_score,
            val_loss,
            precision,
            recall,
            f1,
        ) = evaluate(
            model,
            val_loader,
            'cpu',
            amp,
            batch_size,
            loss_function=loss_function,
            skip=skip_eval,
        )
        torch.cuda.empty_cache()
        # get available devices
        torch.cuda.device_count()
        torch.set_grad_enabled(True)
        model.train()
        model.to(device=device)
        epoch_loss = 0
        # batch accumulation parameter
        accum_loss = 0
        accum_iter = 16
        if accum_iter < batch_size:
            accum_iter = batch_size
        train_len = len(train_loader)
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for i, data in enumerate(train_loader):
                images, true_masks = data

                assert images.shape[1] == model.n_channels, (
                    f'Network has been defined with {model.n_channels} input channels, '
                    f'but loaded images have {images.shape[1]} channels. Please check that '
                    'the images are loaded correctly.'
                )

                images = images.to(
                    device=device,
                    dtype=torch.float16,
                    memory_format=torch.channels_last,
                )
                images.requires_grad = True
                true_masks = true_masks.to(
                    device=device,
                    dtype=torch.long,
                )

                with torch.autocast(
                    device.type if device.type != 'mps' else 'cpu', enabled=amp
                ):
                    if use_checkpointing is True:
                        masks_pred = checkpoint(model, images)
                    else:
                        masks_pred = model(images)
                    if loss_function == 'bce':
                        criterion = nn.BCEWithLogitsLoss(
                            pos_weight=torch.tensor([pos_weight]).to(device),
                        )
                    elif loss_function == 'ft':
                        criterion = FocalTverskyLoss(alpha=0.75, beta=0.25, gamma=1)
                    else:
                        raise NotImplementedError
                    loss = criterion(
                        masks_pred,
                        true_masks.float(),
                    )
                del masks_pred
                # normalize loss to account for batch accumulation
                loss = loss / accum_iter
                loss.backward()
                accum_loss += loss.item()
                # weights update
                if ((i + 1) % accum_iter == 0) or (i + 1 == train_len):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clipping
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    with warmup_scheduler.dampening():
                        lr_scheduler.step()
                    experiment.log(
                        {
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'accum loss': accum_loss,
                            'epoch': epoch,
                        }
                    )
                    accum_loss = 0

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log(
                    {
                        'train loss': epoch_loss / (i + 1),
                        'step': global_step,
                        'epoch': epoch,
                    }
                )
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                del loss

                # Evaluation round
                division_step = n_train // (5 * batch_size)
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(
                                    value.data.cpu()
                                )
                            try:
                                if (
                                    value is not None
                                    and value.grad is not None
                                    and not (
                                        torch.isinf(value.grad)
                                        | torch.isnan(value.grad)
                                    ).any()
                                ):
                                    histograms['Gradients/' + tag] = wandb.Histogram(
                                        value.grad.data.cpu()
                                    )
                            except Exception as e:
                                print(e)
                                pass
                        (
                            val_dice_score,
                            val_auc_score,
                            val_loss,
                            precision,
                            recall,
                            f1,
                        ) = evaluate(
                            model,
                            val_loader,
                            device,
                            amp,
                            batch_size,
                            loss_function=loss_function,
                            skip=skip_eval,
                        )
                        logging.info('Validation Dice score: %s', val_dice_score)
                        logging.info('Validation AUC score: %s', val_auc_score)
                        logging.info('Validation Loss: %s', val_loss)
                        logging.info('Validation Precision: %s', precision)
                        logging.info('Validation Recall: %s', recall)
                        logging.info('Validation F1: %s', f1)
                        try:
                            experiment.log(
                                {
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation Dice': val_dice_score,
                                    'validation AUC': val_auc_score,
                                    'validation loss': val_loss,
                                    'validation precision': precision,
                                    'validation recall': recall,
                                    'validation f1': f1,
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms,
                                }
                            )
                        except Exception as e:
                            print(e)
                            logging.info(f'Wandb logging failed {e}')
                            pass

                        if save_checkpoint:
                            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                            state_dict = model.state_dict()
                            state_dict['mask_values'] = train_set.mask_values
                            torch.save(
                                {
                                    'model_state_dict': state_dict,
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'epoch': epoch,
                                    'validation_auc': val_auc_score,
                                    'input_features': limit_features,
                                },
                                f'{str(dir_checkpoint)}/checkpoint_epoch_{epoch+starting_epoch}_{val_auc_score}.pth',
                            )
                            experiment.log_artifact(
                                Artifact(
                                    f'checkpoint_epoch{epoch+starting_epoch}_{val_auc_score}.pth',
                                    type='model',
                                    metadata=dict(
                                        model_type=model.__class__.__name__,
                                        input_features=limit_features,
                                        starting_epoch=epoch + starting_epoch,
                                        val_auc_score=val_auc_score,
                                    ),
                                )
                            )
                            del histograms
                            del state_dict
                            torch.cuda.empty_cache()
                            gc.collect()
                            logging.info(f'Checkpoint {epoch+starting_epoch} saved!')


@torch.inference_mode()
def evaluate(
    net,
    dataloader,
    device,
    batch_size,
    pos_weight=3.0,
    loss_function='bce',
    n_val=1877,
    skip=False,
    save_predictions=False,
):
    """
    Function to evaluate the model on the validation set. It computes the Dice
    score and the AUC score.

    Args:
        net (torch.nn.Module): the neural network
        dataloader (torch.utils.data.DataLoader): validation set data loader
        device (torch.device): device used for training
        amp (bool): whether to use mixed precision or not
        batch_size (int): batch size
        pos_weight (float): weight for positive class in BCE loss, default is 3.0
        loss_function (str): loss function to use, default is BCE
    Returns:
        dice_score (float): Dice score
        auc_score (float): AUC score
        val_loss (float): validation loss
        precision (float): precision
        recall (float): recall
        f1 (float): weighted f1 score
        n_val (int): number of validation samples
    """
    save_dir = Path(f'predictions_{POS_THRESHOLD}')
    os.makedirs(save_dir, exist_ok=True)
    net.eval()
    net.to(device=device)
    torch.set_grad_enabled(False)
    num_val_batches = n_val // batch_size
    val_loss = 0
    dice_score = 0
    auc_score = 0
    precision = 0
    recall = 0
    f1 = 0

    if skip:
        net.train()
        return dice_score, auc_score, val_loss, precision, recall, f1

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=False):
        for i, batch in tqdm(
            enumerate(dataloader),
            total=num_val_batches,
            desc='Validation round',
            unit='batch',
            leave=False,
        ):
            image, mask_true = batch

            # move images and labels to correct device and type
            image = image.to(
                device=device, dtype=torch.float32, memory_format=torch.channels_last
            )
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            if net.n_classes == 1:
                mask_true_np = mask_true.cpu().numpy()
                mask_pred_np = mask_pred.cpu().numpy()
                # compute the AUC score from logits
                auc_score += get_auc(
                    mask_true.cpu().numpy(), mask_pred.cpu().numpy(), with_logits=True
                )
                if net.__class__.__name__ == 'FireRevColNet':
                    mask_pred = (F.sigmoid(mask_pred) > REVCOL_THRESHOLD).float()
                else:
                    mask_pred = (F.sigmoid(mask_pred) > POS_THRESHOLD).float()
                mask_pred_np = mask_pred.cpu().numpy()
                precision += get_precision(mask_true_np, mask_pred_np)
                recall += get_recall(mask_true_np, mask_pred_np)
                # p, r = get_precision_recall(mask_pred_np, mask_true_np)
                f1 += get_weighted_f1(mask_true.cpu(), mask_pred.cpu())
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                if loss_function == 'bce':
                    criterion = nn.BCEWithLogitsLoss(
                        pos_weight=torch.tensor([pos_weight]).to(device),
                    )
                elif loss_function == 'ft':
                    criterion = FocalTverskyLoss(alpha=0.75, beta=0.25, gamma=1)
                else:
                    raise NotImplementedError

                val_loss += criterion(
                    mask_pred,
                    mask_true.float(),
                )
                if save_predictions:
                    # save the predicted masks
                    mask_pred = mask_pred.squeeze(1).cpu().numpy()
                    mask_true = mask_true.squeeze(1).cpu().numpy()
                    for j in range(mask_pred.shape[0]):
                        # Only save a random 100 positive masks
                        if (mask_pred[j].max() > 0) and (random.random() < 0.1):
                            plt.imsave(
                                save_dir / f'{i * batch_size + j}_pred.png',
                                mask_pred[j],
                                cmap='OrRd',
                            )
                            # save the true masks
                            plt.imsave(
                                save_dir / f'{i * batch_size + j}_true.png',
                                mask_true[j],
                                cmap='OrRd',
                            )

    net.train()
    return (
        dice_score / max(num_val_batches, 1),
        auc_score / max(num_val_batches, 1),
        val_loss / max(num_val_batches, 1),
        precision / max(num_val_batches, 1),
        recall / max(num_val_batches, 1),
        f1 / max(num_val_batches, 1),
    )


def load_checkpoint(checkpoint_file: str) -> tuple:
    """
    Loads model and optimizer state from a checkpoint file. This is useful when
    you want to test your model on a test dataset without having to train it
    again.

    Args:
        checkpoint_file (str): path to checkpoint file to be loaded
    Returns:
        model_state_dict (OrderedDict): model's weights
        optimizer_state_dict (OrderedDict): optimizer's state_dict
        epoch (int): epoch number from which training resumes
        validation_auc (float): validation auc score
        input_features (list): list of input features
    """
    state_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    model_state_dict = state_dict['model_state_dict']
    optimizer_state_dict = state_dict['optimizer_state_dict']
    epoch = state_dict['epoch']
    validation_auc = state_dict['validation_auc']
    input_features = (
        state_dict['input_features'] if 'input_features' in state_dict else None
    )
    del model_state_dict['mask_values']
    return model_state_dict, optimizer_state_dict, epoch, validation_auc, input_features
