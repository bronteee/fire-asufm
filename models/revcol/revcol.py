# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
from models.revcol.modules import (
    ConvNextBlock,
    Decoder,
    LayerNorm,
    SimDecoder,
    UpSampleConvnext,
)
import torch.distributed as dist
from models.revcol.revcol_function import ReverseFunction
from timm.models.layers import trunc_normal_


class Fusion(nn.Module):
    def __init__(self, level, channels, first_col) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = (
            nn.Sequential(
                nn.Conv2d(
                    channels[level - 1], channels[level], kernel_size=2, stride=2
                ),
                LayerNorm(channels[level], eps=1e-6, data_format="channels_first"),
            )
            if level in [1, 2, 3]
            else nn.Identity()
        )
        if not first_col:
            self.up = (
                UpSampleConvnext(1, channels[level + 1], channels[level])
                if level in [0, 1, 2]
                else nn.Identity()
            )

    def forward(self, *args):
        c_down, c_up = args

        if self.first_col:
            x = self.down(c_down)
            return x

        if self.level == 3:
            x = self.down(c_down)
        else:
            x = self.up(c_up) + self.down(c_down)
        return x


class Level(nn.Module):
    def __init__(
        self, level, channels, layers, kernel_size, first_col, dp_rate=0.0
    ) -> None:
        super().__init__()
        countlayer = sum(layers[:level])
        expansion = 4
        self.fusion = Fusion(level, channels, first_col)
        modules = [
            ConvNextBlock(
                channels[level],
                expansion * channels[level],
                channels[level],
                kernel_size=kernel_size,
                layer_scale_init_value=1e-6,
                drop_path=dp_rate[countlayer + i],
            )
            for i in range(layers[level])
        ]
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x = self.fusion(*args)
        x = self.blocks(x)
        return x


class SubNet(nn.Module):
    def __init__(
        self, channels, layers, kernel_size, first_col, dp_rates, save_memory
    ) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        self.save_memory = save_memory
        self.alpha0 = (
            nn.Parameter(
                shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                requires_grad=True,
            )
            if shortcut_scale_init_value > 0
            else None
        )
        self.alpha1 = (
            nn.Parameter(
                shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                requires_grad=True,
            )
            if shortcut_scale_init_value > 0
            else None
        )
        self.alpha2 = (
            nn.Parameter(
                shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                requires_grad=True,
            )
            if shortcut_scale_init_value > 0
            else None
        )
        self.alpha3 = (
            nn.Parameter(
                shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                requires_grad=True,
            )
            if shortcut_scale_init_value > 0
            else None
        )

        self.level0 = Level(0, channels, layers, kernel_size, first_col, dp_rates)

        self.level1 = Level(1, channels, layers, kernel_size, first_col, dp_rates)

        self.level2 = Level(2, channels, layers, kernel_size, first_col, dp_rates)

        self.level3 = Level(3, channels, layers, kernel_size, first_col, dp_rates)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):
        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_channels, eps=1e-6),  # final norm layer
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FireRevColNet(nn.Module):
    def __init__(
        self,
        n_channels=12,
        n_classes=1,
        channels=[32, 64, 96, 128],
        layers=[2, 3, 6, 3],
        num_subnet=5,
        kernel_size=3,
        num_classes=1000,
        drop_path=0.0,
        save_memory=True,
        inter_supv=True,
        head_init_scale=None,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.in_chans = n_channels
        self.n_classes = n_classes
        self.num_classes = n_classes
        self.num_subnet = num_subnet
        self.inter_supv = inter_supv
        self.channels = channels
        self.layers = layers

        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, channels[0], kernel_size=4, stride=4),
            LayerNorm(channels[0], eps=1e-6, data_format="channels_first"),
        )

        dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(layers))]
        for i in range(num_subnet):
            first_col = True if i == 0 else False
            self.add_module(
                f'subnet{str(i)}',
                SubNet(
                    channels,
                    layers,
                    kernel_size,
                    first_col,
                    dp_rates=dp_rate,
                    save_memory=save_memory,
                ),
            )

        if not inter_supv:
            self.cls = Classifier(in_channels=channels[-1], num_classes=num_classes)
        else:
            self.cls_blocks = nn.ModuleList(
                [
                    Classifier(in_channels=channels[-1], num_classes=num_classes)
                    for _ in range(4)
                ]
            )
            if num_classes <= 1000:
                channels.reverse()
                self.decoder_blocks = nn.ModuleList(
                    [
                        Decoder(
                            depth=[1, 1, 1, 1],
                            dim=channels,
                            block_type=ConvNextBlock,
                            kernel_size=3,
                        )
                        for _ in range(3)
                    ]
                )
            else:
                self.decoder_blocks = nn.ModuleList(
                    [
                        SimDecoder(in_channel=channels[-1], encoder_stride=32)
                        for _ in range(3)
                    ]
                )

        self.apply(self._init_weights)

        if head_init_scale:
            print(f'Head_init_scale: {head_init_scale}')
            self.cls.classifier._modules['1'].weight.data.mul_(head_init_scale)
            self.cls.classifier._modules['1'].bias.data.mul_(head_init_scale)

    def forward(self, x):
        if self.inter_supv:
            return self._forward_intermediate_supervision(x)
        else:
            c0, c1, c2, c3 = 0, 0, 0, 0
            x = self.stem(x)
            for i in range(self.num_subnet):
                c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2, c3)
            return [self.cls(c3)], None

    def _forward_intermediate_supervision(self, x):
        x_cls_out = []
        x_img_out = []
        c0, c1, c2, c3 = 0, 0, 0, 0
        interval = self.num_subnet // 4

        x = self.stem(x)
        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2, c3)
            if (i + 1) % interval == 0:
                x_cls_out.append(self.cls_blocks[i // interval](c3))
                if i != self.num_subnet - 1:
                    x_img_out.append(self.decoder_blocks[i // interval](c3))

        return x_cls_out, x_img_out

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)


##-------------------------------------- Tiny -----------------------------------------


def revcol_tiny(
    save_memory, inter_supv=True, drop_path=0.1, num_classes=1000, kernel_size=3
):
    channels = [64, 128, 256, 512]
    layers = [2, 2, 4, 2]
    num_subnet = 4
    return FireRevColNet(
        channels=channels,
        layers=layers,
        num_subnet=num_subnet,
        num_classes=num_classes,
        drop_path=drop_path,
        save_memory=save_memory,
        inter_supv=inter_supv,
        kernel_size=kernel_size,
    )


##-------------------------------------- Small -----------------------------------------


def revcol_small(
    save_memory, inter_supv=True, drop_path=0.3, num_classes=1000, kernel_size=3
):
    channels = [64, 128, 256, 512]
    layers = [2, 2, 4, 2]
    num_subnet = 8
    return FireRevColNet(
        channels,
        layers,
        num_subnet,
        num_classes=num_classes,
        drop_path=drop_path,
        save_memory=save_memory,
        inter_supv=inter_supv,
        kernel_size=kernel_size,
    )


##-------------------------------------- Base -----------------------------------------


def revcol_base(
    save_memory,
    inter_supv=True,
    drop_path=0.4,
    num_classes=1000,
    kernel_size=3,
    head_init_scale=None,
):
    channels = [72, 144, 288, 576]
    layers = [1, 1, 3, 2]
    num_subnet = 16
    return FireRevColNet(
        channels,
        layers,
        num_subnet,
        num_classes=num_classes,
        drop_path=drop_path,
        save_memory=save_memory,
        inter_supv=inter_supv,
        head_init_scale=head_init_scale,
        kernel_size=kernel_size,
    )


##-------------------------------------- Large -----------------------------------------


def revcol_large(
    save_memory,
    inter_supv=True,
    drop_path=0.5,
    num_classes=1000,
    kernel_size=3,
    head_init_scale=None,
):
    channels = [128, 256, 512, 1024]
    layers = [1, 2, 6, 2]
    num_subnet = 8
    return FireRevColNet(
        channels,
        layers,
        num_subnet,
        num_classes=num_classes,
        drop_path=drop_path,
        save_memory=save_memory,
        inter_supv=inter_supv,
        head_init_scale=head_init_scale,
        kernel_size=kernel_size,
    )


##--------------------------------------Extra-Large -----------------------------------------
def revcol_xlarge(
    save_memory,
    inter_supv=True,
    drop_path=0.5,
    num_classes=1000,
    kernel_size=3,
    head_init_scale=None,
):
    channels = [224, 448, 896, 1792]
    layers = [1, 2, 6, 2]
    num_subnet = 8
    return FireRevColNet(
        channels,
        layers,
        num_subnet,
        num_classes=num_classes,
        drop_path=drop_path,
        save_memory=save_memory,
        inter_supv=inter_supv,
        head_init_scale=head_init_scale,
        kernel_size=kernel_size,
    )


# Based on https://github.com/megvii-research/RevCol/blob/main/loss.py
def compound_loss(
    coe,
    output_feature,
    image: Tensor,
    criterion_bce,
):
    f_coe, c_coe = coe
    image.clamp_(0.01, 0.99)
    multi_loss = []
    for i, feature in enumerate(output_feature):
        ratio_f = 1 - i / len(output_feature)
        # ratio_c = (i + 1) / (len(output_label))

        ihx = criterion_bce(feature, image) * ratio_f * f_coe
        # ihy = criterion_ce(output_label[i], targets) * ratio_c * c_coe
        # if dist.get_rank() == 0:
        #     print(f'ihx: {ihx}, ihy: {ihy}')
        multi_loss.append(ihx)
        # feature_loss.append(torch.dist(output_feature[i], teacher_feature) *  feature_coe)
    # multi_loss.append(criterion_ce(output_label[-1], targets))
    # print(feature_loss)
    loss = torch.sum(torch.stack(multi_loss), dim=0)
    # +torch.mean(torch.stack(feature_loss), dim=0)
    return loss


def compound_prediction(coe, pred_masks, concensus='mean'):
    f_coe, c_coe = coe
    decisions = []
    # for i, pred in enumerate(pred_masks):
    #     # ratio_f = 1 - i / len(pred_masks)
    #     # logits = pred * ratio_f * f_coe
    #     logits = pred
    #     decisions.append(logits)
    # if concensus == 'mean':
    #     final_pred = torch.mean(torch.stack(decisions), dim=0)
    # elif concensus == 'sum':
    #     final_pred = torch.sum(torch.stack(decisions), dim=0)
    final_pred = pred_masks[0] * f_coe
    return final_pred


if __name__ == '__main__':
    # Test model with [B, 12, 64, 64] input
    model = revcol_tiny(
        save_memory=True, inter_supv=True, drop_path=0.1, num_classes=1, kernel_size=3
    )
    x = torch.randn(1, 12, 64, 64)
    target_mask = torch.randn(1, 1, 64, 64)
    y = model(x)
    compounded_loss = compound_loss(
        coe=(1, 1),
        output_feature=y[1],
        image=target_mask,
        criterion_bce=nn.BCEWithLogitsLoss(),
    )
    print(compounded_loss)
    pred = compound_prediction(coe=(1, 1), pred_masks=y[1], concensus='max')
    print(pred.shape)
    print(pred)
