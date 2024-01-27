"""
Bronte Sihan Li, 2024

Based on the work of:
https://github.com/google-research/google-research/tree/master/simulation_research/next_day_wildfire_spread
https://github.com/Jo-dsa/SemanticSeg/tree/master

"""

import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from torch import Tensor
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics import F1Score
from torchmetrics.classification import Dice
from torchmetrics import PrecisionRecallCurve


def weighted_cross_entropy_with_logits_with_masked_class(pos_weight=1.0):
    """Wrapper function for masked weighted cross-entropy with logits.

    This loss function ignores the classes with negative class id.

    Args:
      pos_weight: A coefficient to use on the positive examples.

    Returns:
      A weighted cross-entropy with logits loss function that ignores classes
      with negative class id.
    """

    def masked_weighted_cross_entropy_with_logits(y_true, logits):
        y_true = tf.cast(y_true, tf.float32)
        logits = tf.cast(logits, tf.float32)
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true, logits=logits, pos_weight=pos_weight
        )
        return tf.reduce_mean(loss, 0).numpy()

    return masked_weighted_cross_entropy_with_logits


def calculate_pos_weights(class_counts, data):
    pos_weights = np.ones_like(class_counts)
    neg_counts = [len(data) - pos_count for pos_count in class_counts]
    for cdx, pos_count, neg_count in enumerate(zip(class_counts, neg_counts)):
        pos_weights[cdx] = neg_count / (pos_count + 1e-5)

    return torch.as_tensor(pos_weights, dtype=torch.float)


def multiclass_dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    # Average of Dice coefficient for all classes
    return dice_coeff(
        input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon
    )


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


class CustomDiceMetric(Dice):
    def __init__(self, ignore=-1, **kwargs):
        super().__init__(**kwargs)
        self.ignore = ignore

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Create a mask for the values to be ignored
        ignore_mask = target != self.ignore

        # Apply the mask
        preds = preds[ignore_mask]
        target = target[ignore_mask]

        # Update the metric with the masked values
        super().update(preds, target)


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    dice_metric = CustomDiceMetric()
    dice_metric.update(preds=input, target=target)

    return dice_metric.compute()


def iou(
    pred: Tensor,
    target: Tensor,
):
    # Average of IoU for all batches, or for a single mask
    assert (
        pred.size() == target.size()
    ), "'input' and 'target' must have the same shape, got {} and {}".format(
        pred.size(), target.size()
    )

    metric = BinaryJaccardIndex()
    return metric(pred, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(ignore_index=-1)(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, preds, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        preds = torch.sigmoid(preds)

        # flatten label and prediction tensors
        preds = preds.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (preds * targets).sum()
        FP = ((1 - targets) * preds).sum()
        FN = (targets * (1 - preds)).sum()

        tversky_index = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        focal_tversky = torch.mean((1 - tversky_index) ** self.gamma)

        return focal_tversky


def tversky_index(yhat, ytrue, alpha=0.3, beta=0.7, epsilon=1e-6):
    """
    Computes Tversky index

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        alpha (Float): weight for False positive
        beta (Float): weight for False negative
                    `` alpha and beta control the magnitude of penalties and should sum to 1``
        epsilon (Float): smoothing value to avoid division by 0
    output:
        tversky index value
    """
    TP = torch.sum(yhat * ytrue, (1, 2, 3))
    FP = torch.sum((1.0 - ytrue) * yhat, (1, 2, 3))
    FN = torch.sum((1.0 - yhat) * ytrue, (1, 2, 3))

    return TP / (TP + alpha * FP + beta * FN + epsilon)


def tversky_loss(yhat, ytrue):
    """
    Computes tversky loss given tversky index

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
    output:
        tversky loss value with `mean` reduction
    """
    return torch.mean(1 - tversky_index(yhat, ytrue))


def tversky_focal_loss(yhat, ytrue, alpha=0.7, beta=0.3, gamma=0.75):
    """
    Computes tversky focal loss for highly umbalanced data
    https://arxiv.org/pdf/1810.07842.pdf

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        alpha (Float): weight for False positive
        beta (Float): weight for False negative
                    `` alpha and beta control the magnitude of penalties and should sum to 1``
        gamma (Float): focal parameter
                    ``control the balance between easy background and hard ROI training examples``
    output:
        tversky focal loss value with `mean` reduction
    """

    return torch.mean(torch.pow(1 - tversky_index(yhat, ytrue, alpha, beta), gamma))


def focal_loss(yhat, ytrue, alpha=0.75, gamma=2):
    """
    Computes α-balanced focal loss from FAIR
    https://arxiv.org/pdf/1708.02002v2.pdf

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        alpha (Float): weight to balance Cross entropy value
        gamma (Float): focal parameter
    output:
        loss value with `mean` reduction
    """

    # compute the actual focal loss
    focal = -alpha * torch.pow(1.0 - yhat, gamma) * torch.log(yhat)
    f_loss = torch.sum(ytrue * focal, dim=1)

    return torch.mean(f_loss)


class AUCWithMaskedClass(tf.keras.metrics.AUC):
    """Computes AUC while ignoring class with id equal to `-1`.

    Assumes binary `{0, 1}` classes with a masked `{-1}` class.
    """

    def __init__(self, with_logits=False, **kwargs):
        super(AUCWithMaskedClass, self).__init__(**kwargs)
        self.with_logits = with_logits

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates metric statistics.

        `y_true` and `y_pred` should have the same shape.

        Args:
          y_true: Ground truth values.
          y_pred: Predicted values.
          sample_weight: Input value is ignored. Parameter present to match
            signature with parent class where mask `{-1}` is the sample weight.
        Returns: `None`
        """
        if self.with_logits:
            y_pred = tf.math.sigmoid(y_pred)
        mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
        super(AUCWithMaskedClass, self).update_state(y_true, y_pred, sample_weight=mask)


class PrecisionWithMaskedClass(tf.keras.metrics.Precision):
    """Computes precision while ignoring class with id equal to `-1`.

    Assumes binary `{0, 1}` classes with a masked `{-1}` class.
    """

    def __init__(self, with_logits=False, **kwargs):
        super(PrecisionWithMaskedClass, self).__init__(**kwargs)
        self.with_logits = with_logits

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates metric statistics.

        `y_true` and `y_pred` should have the same shape.

        Args:
          y_true: Ground truth values.
          y_pred: Predicted values.
          sample_weight: Input value is ignored. Parameter present to match
            signature with parent class where mask `{-1}` is the sample weight.
        Returns: `None`
        """
        if self.with_logits:
            y_pred = tf.math.sigmoid(y_pred)
        mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
        super(PrecisionWithMaskedClass, self).update_state(
            y_true, y_pred, sample_weight=mask
        )


class RecallWithMaskedClass(tf.keras.metrics.Recall):
    """Computes recall while ignoring class with id equal to `-1`.

    Assumes binary `{0, 1}` classes with a masked `{-1}` class.
    """

    def __init__(self, with_logits=False, **kwargs):
        super(RecallWithMaskedClass, self).__init__(**kwargs)
        self.with_logits = with_logits

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates metric statistics.

        `y_true` and `y_pred` should have the same shape.

        Args:
          y_true: Ground truth values.
          y_pred: Predicted values.
          sample_weight: Input value is ignored. Parameter present to match
            signature with parent class where mask `{-1}` is the sample weight.
        Returns: `None`
        """
        if self.with_logits:
            y_pred = tf.math.sigmoid(y_pred)
        mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
        super(RecallWithMaskedClass, self).update_state(
            y_true, y_pred, sample_weight=mask
        )


def get_auc(y_true: np.ndarray, y_pred: np.ndarray, with_logits=False):
    """Computes AUC while ignoring class with id equal to `-1`.
    Assumes binary `{0, 1}` classes with a masked `{-1}` class.
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        AUC score.
    """
    metric = AUCWithMaskedClass(with_logits=with_logits, curve='PR')
    metric.update_state(y_true, y_pred)
    return metric.result().numpy()


def get_recall(
    y_true: np.ndarray, y_pred: np.ndarray, with_logits=False, pos_only=False
):
    """Computes recall while ignoring class with id equal to `-1`.
    Assumes binary `{0, 1}` classes with a masked `{-1}` class.
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Recall score.
    """
    metric = RecallWithMaskedClass(
        with_logits=with_logits,
    )

    metric.update_state(y_true, y_pred)
    return metric.result().numpy()


def get_precision(
    y_true: np.ndarray, y_pred: np.ndarray, with_logits=False, pos_only=False
):
    """Computes precision while ignoring class with id equal to `-1`.
    Assumes binary `{0, 1}` classes with a masked `{-1}` class.
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Precision score.
    """
    metric = PrecisionWithMaskedClass(
        with_logits=with_logits,
    )
    metric.update_state(
        y_true,
        y_pred,
    )
    return metric.result().numpy()


def get_weighted_f1(
    y_true,
    y_pred,
):
    return F1Score(task='binary', num_classes=2, average='weighted', ignore_index=-1)(
        y_pred, y_true
    )


def get_precision_recall(y_pred, y_true):
    pr_curve = PrecisionRecallCurve(task='binary', num_classes=1, ignore_index=-1)
    precision, recall, thresholds = pr_curve(y_pred, y_true)
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'thresholds: {thresholds}')
    return precision, recall, thresholds
