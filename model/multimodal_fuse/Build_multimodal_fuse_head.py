from model.multimodal_fuse.Head_layers import *


class Build_multimodal_fuse_head(nn.Module):
    """
    Build multimodal_fuse_head

    Args:
        n_branch (int): Number of branches.
        in_chans (int, tuple[int]): Number of channels for input images in each branch. Default: (3, 3, 3, 3)
        out_chans (int): Number of output image channels. Default: 36
        n_group (int): Number of groups.
        patch_size (int): Branch self-attention patch size. Default: 4
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: False
        chan_ratio (int): Scaling ratio of 'CBAM' or 'SE' channel attention. Default: 16
        n_heads (int): Number of SA channel attention heads. Default: 4
        fuse_type: AMM method.

    Return shape: (b c h w)
    """

    def __init__(self,
                 n_branch,
                 in_chans=(3, 3, 3, 3),
                 out_chans=36,
                 n_group=3,
                 use_pos=True,
                 patch_size=4,
                 attn_drop=0.1,
                 qkv_bias=False,
                 offset_scale=8,
                 chan_ratio=16,
                 chan_attn_type='SE',
                 n_heads=2,
                 fuse_type=None,
                 embed_dim=None):
        super().__init__()
        in_chans = in_chans if isinstance(in_chans, tuple) else tuple([in_chans for _ in range(n_branch)])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fuse_type = fuse_type
        self.use_pos = use_pos

        self.MSEs = nn.ModuleList([
            MSE(in_chans=in_chans[i],
                out_chans=out_chans,
                n_group=n_group,
                use_pos=use_pos,
                channel_attn_type=chan_attn_type,
                ratio=chan_ratio)
            for i in range(n_branch)])

        if use_pos:
            ang_table = [ang for ang in range(0, 136, 135 // n_branch)]
            self.pos = [nn.Parameter(torch.tensor([np.cos(ang_table[i] * np.pi / 180)], dtype=torch.float32))
                        for i in range(n_branch)]

        smooth_chans = int(out_chans * n_branch)
        self.smooth = nn.Sequential(
            nn.Conv2d(smooth_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU()
        )

        if self.fuse_type is None:
            self.fuse_proj = AMM(in_chans=smooth_chans,
                                 out_chans=embed_dim,
                                 n_branch=n_branch,
                                 n_heads=n_heads,
                                 offset_scale=offset_scale,
                                 patch_size=patch_size,
                                 fuse_drop=attn_drop,
                                 qkv_bias=qkv_bias)
        else:
            self.fuse_proj = nn.Identity()

    def forward(self, x):
        x = x if isinstance(x, tuple and list) else tuple([x])
        fuse = []
        for i, layer in enumerate(self.MSEs):
            x_branch = layer(x[i], self.pos[i].to(device=self.device) if self.use_pos else None)
            fuse.append(x_branch)
        x = self.fuse_proj(torch.cat(fuse, dim=1))

        if self.fuse_type is not None:
            x = self.smooth(x)
            return x
        else:
            de_x = self.smooth(x[1])
            return x[0], de_x
