import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# ---------------- ViT blocks ----------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)



class PoseGlobalHead(nn.Module):
    def __init__(self, in_ch: int, nc: int, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.nc = int(nc)
        self.in_ch = int(in_ch)

        mlp = [
            nn.Linear(in_ch, hidden),
            nn.SiLU(),
        ]
        if dropout > 0:
            mlp.append(nn.Dropout(dropout))
        mlp += [
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        ]
        if dropout > 0:
            mlp.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*mlp)

        self.fc_center = nn.Linear(hidden, 3)   # center_norm
        self.fc_normal = nn.Linear(hidden, 3)   # normal raw
        self.fc_cls = nn.Linear(hidden, self.nc)

        nn.init.zeros_(self.fc_center.bias)
        nn.init.zeros_(self.fc_normal.bias)

    def forward(self, global_vec):
        feat = self.mlp(global_vec)
        center_norm = self.fc_center(feat)
        normal_raw = self.fc_normal(feat)
        cls_logits = self.fc_cls(feat)
        return center_norm, normal_raw, cls_logits


# ---------------- Pure Global ViT Pose ----------------
class ViTGlobalPose(nn.Module):
    """
    global-only:
      input:  images [B,3,H,W]
      output:
        train: {"center_norm", "normal", "cls_logits"}
        eval : {"center_m", "normal", "cls_prob", "cls_logits"}
    """
    def __init__(
        self,
        *,
        image_size=640,
        patch_size=32,
        num_classes,
        dim=512,
        depth=8,
        heads=8,
        mlp_dim=1024,
        pool='cls',
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        head_hidden=512,
        head_dropout=0.0,
    ):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        assert pool in {'cls', 'mean'}, \
            'pool type must be either cls or mean'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1=patch_height, p2=patch_width
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.head = PoseGlobalHead(
            in_ch=dim,
            nc=num_classes,
            hidden=head_hidden,
            dropout=head_dropout,
        )


        self.register_buffer("center_mean", torch.zeros(1, 3))
        self.register_buffer("center_std", torch.ones(1, 3))
        self.denorm_inference = True

    @torch.no_grad()
    def set_center_stats(self, mean_xyz, std_xyz, eps=1e-6, denorm_inference=True):
        m = torch.as_tensor(mean_xyz, dtype=torch.float32).view(1, 3)
        s = torch.as_tensor(std_xyz, dtype=torch.float32).view(1, 3).clamp_min(eps)
        self.center_mean.copy_(m)
        self.center_std.copy_(s)
        self.denorm_inference = bool(denorm_inference)

    def forward(self, images: torch.Tensor):
        x = self.to_patch_embedding(images)   # [B, N, dim]
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # global_vec
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        global_vec = self.to_latent(x)

        center_norm, normal_raw, cls_logits = self.head(global_vec)

        normal = F.normalize(normal_raw, dim=1, eps=1e-6)

        if self.training:
            return {
                "center_norm": center_norm,
                "normal": normal,
                "cls_logits": cls_logits,
            }

        if self.denorm_inference:
            center_m = center_norm * self.center_std + self.center_mean
        else:
            center_m = center_norm

        cls_prob = F.softmax(cls_logits, dim=1)
        return {
            "center_m": center_m,
            "normal": normal,
            "cls_prob": cls_prob,
            "cls_logits": cls_logits,
        }

    def fuse(self):
        return self


def vit_global_pose_b(
    num_classes: int,
    img_size: int = 640,
    patch_size: int = 32,
):
    return ViTGlobalPose(
        image_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=512,
        depth=8,
        heads=8,
        mlp_dim=1024,
        pool='cls',
        channels=3,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1,
        head_hidden=512,   
        head_dropout=0.0,  
    )


def vit_global_pose_l(
    num_classes: int,
    img_size: int = 640,
    patch_size: int = 32,
):
    return ViTGlobalPose(
        image_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=1536,
        pool='cls',
        channels=3,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1,
        head_hidden=512,
        head_dropout=0.0,
    )