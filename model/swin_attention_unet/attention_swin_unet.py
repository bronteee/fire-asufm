import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .embedding import PatchEmbed
from .encoder import Encoder
from .decoder import Decoder


class ASUFM(nn.Module):
    r"""Ateention Swin Unet with Focal Modulation
    we use A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
      https://arxiv.org/pdf/2103.14030
    and edited some parts.
    """

    def __init__(self, config, num_classes=1):
        super().__init__()
        self.n_channels = config.in_chans
        self.n_classes = num_classes
        self.img_size = config.image_size
        self.patch_size = config.patch_size
        self.in_chans = config.patch_size
        self.embed_dim = config.embed_dim
        self.depths = config.depths
        self.num_heads = config.num_heads
        self.window_size = config.window_size
        self.mlp_ratio = config.mlp_ratio
        self.qkv_bias = config.qkv_bias
        self.qk_scale = config.qk_scale
        self.drop_rate = config.drop_rate
        self.drop_path_rate = config.drop_path_rate
        self.focal = config.focal
        self.attn_drop_rate = 0
        self.ape = config.ape
        self.patch_norm = config.patch_norm
        self.use_checkpoint = config.use_checkpoint
        self.num_classes = num_classes
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(self.embed_dim * 2)
        self.mode = config.mode
        self.skip_num = config.skip_num
        self.operation = config.operationaddatten
        self.add_attention = config.spatial_attention
        self.final_upsample = "expand_first"
        self.norm_layer = nn.LayerNorm

        # Build embedding
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=nn.LayerNorm if self.patch_norm else None,
        )
        self.num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        # Build encoder
        self.encoder = Encoder(
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            norm_layer=self.norm_layer,
            use_checkpoint=self.use_checkpoint,
            num_layers=self.num_layers,
            img_size=self.img_size,
            ape=False,
            num_patches=self.num_patches,
            patch_size=4,
            in_chans=config.in_chans,
            drop_path_rate=0.1,
            patch_embed=self.patch_embed,
            args=config,
            focal=self.focal,
        )
        # Build decoder
        self.decoder = Decoder(
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            img_size=self.img_size,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            norm_layer=self.norm_layer,
            use_checkpoint=self.use_checkpoint,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            patch_embed=self.patch_embed,
            patch_norm=True,
            final_upsample="expand_first",
            args=config,
            focal=self.focal,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, x_downsample, x_attention_encoder = self.encoder(x)
        x = self.decoder(x, x_downsample, x_attention_encoder)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops
