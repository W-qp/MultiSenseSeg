from model.pipeline.Pipeline_layers import *


class Build_backbone(nn.Module):
    """
    Build feature extraction pipeline

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Number of output channels.
        group_dim (int):  Group dimension. Default: 16
        depths (tuple[int]): Depths of each stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 8.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qk_ratio (float | None):  Scaling ratio of q, k. Default: 3
        down_ratio (int): Amplification ratio of the number of dim after downsampling. Default: 2
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (str): Normalization layer type, use 'BN' or 'LN'. Default: 'BN'
        act_layer (optional): Act layer. Default: nn.ReLU
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        qkv_type (str): Type of qkv, use 'CNN' or 'FC'.  Default: FC
        ffn_type (str): Type of feed forword net, use 'CNN' or 'FC'.  Default: CNN
    """

    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 group_dim=16,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=8,
                 mlp_ratio=4.,
                 qk_ratio=3,
                 down_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer='BN',
                 act_layer=nn.ReLU,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 qkv_type='FC',
                 ffn_type='CNN'):
        super().__init__()
        for i, stage_depth in enumerate(depths):
            assert stage_depth % 2 == 0, f"Stage{i}'s depth must be even, but stage{i}_depth = {stage_depth} !!"
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        if patch_size is not None:
            patch_size = to_2tuple(patch_size)
            self.patch_embed = PatchEmbed_block(patch_size=patch_size, in_chans=in_chans, out_chans=embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build MSEs
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                d=int(embed_dim * 2 ** i_layer),
                group_dim=group_dim,
                depth=depths[i_layer],
                n_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qk_ratio=qk_ratio,
                down_ratio=down_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=Downsampling_block if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                qkv_type=qkv_type,
                ffn_type=ffn_type,
                norm_layer=norm_layer,
                act_layer=act_layer)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = creat_norm_layer('LN', num_features[i_layer], True)
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        if self.patch_size is not None:
            x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep MSEs freezed."""
        super(Build_backbone, self).train(mode)
        self._freeze_stages()


class CNN_Block(nn.Module):
    expansion = 1

    def __init__(self, in_chans, planes, stride=1):
        super(CNN_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_chans != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chans, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class CNN_backbone(nn.Module):
    def __init__(self, chans):
        super(CNN_backbone, self).__init__()
        self.in_planes = chans
        self.layer1 = self._make_layer(chans, 3, stride=1)
        self.layer2 = self._make_layer(chans*2, 4, stride=2)
        self.layer3 = self._make_layer(chans*4, 6, stride=2)
        self.layer4 = self._make_layer(chans*8, 3, stride=2)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(CNN_Block(self.in_planes, planes, stride))
            self.in_planes = planes * CNN_Block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        return feat1, feat2, feat3, feat4
