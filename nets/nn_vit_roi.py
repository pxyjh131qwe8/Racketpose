# nets/nn_roi_pose.py
import math
import torch
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

# 复用现有 Conv / fuse_conv / DarkNet / DarkFPN
from nets.nn_center_and_normal import DarkNet, DarkFPN, Conv, fuse_conv

# ============= ViT backbone (输出 feature map，不输出分类) =============
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = torch.nn.LayerNorm(dim)
        self.attend = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout)

        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(torch.nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViTBackbone(torch.nn.Module):
    """
    输入:  [B,3,H,W]
    输出:  feat_map [B, dim, H/patch, W/patch]
    """
    def __init__(
        self,
        image_size=640,
        patch_size=8,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            "Image size must be divisible by patch size."

        self.image_size = (image_height, image_width)
        self.patch_size = (patch_height, patch_width)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.grid_h = image_height // patch_height
        self.grid_w = image_width // patch_width
        self.dim = dim

        self.to_patch_embedding = torch.nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            torch.nn.LayerNorm(patch_dim),
            torch.nn.Linear(patch_dim, dim),
            torch.nn.LayerNorm(dim),
        )

        self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = torch.nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img: torch.Tensor):
        x = self.to_patch_embedding(img)  # [B, N, dim]
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)             # [B, 1+N, dim]
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)                           # [B, 1+N, dim]
        patch_tokens = x[:, 1:, :]                        # [B, N, dim]

        feat = patch_tokens.view(b, self.grid_h, self.grid_w, self.dim)
        feat = feat.permute(0, 3, 1, 2).contiguous()      # [B, dim, Gh, Gw]
        return feat


# ============= ROI head 不变 =============
class PoseROIHead(torch.nn.Module):
    def __init__(self, in_ch: int, nc: int, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.nc = int(nc)

        self.conv = torch.nn.Sequential(
            Conv(in_ch, in_ch, torch.nn.SiLU(), k=3, p=1),
            Conv(in_ch, in_ch, torch.nn.SiLU(), k=3, p=1),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        mlp = [
            torch.nn.Linear(in_ch, hidden),
            torch.nn.SiLU(),
        ]
        if dropout > 0:
            mlp.append(torch.nn.Dropout(dropout))
        mlp += [
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        ]
        if dropout > 0:
            mlp.append(torch.nn.Dropout(dropout))
        self.mlp = torch.nn.Sequential(*mlp)

        self.fc_center = torch.nn.Linear(hidden, 3)   # center_norm
        self.fc_normal = torch.nn.Linear(hidden, 3)   # normal raw
        self.fc_cls = torch.nn.Linear(hidden, self.nc)

        torch.nn.init.zeros_(self.fc_center.bias)
        torch.nn.init.zeros_(self.fc_normal.bias)

    def forward(self, roi_feat: torch.Tensor):
        x = self.conv(roi_feat)
        x = self.pool(x).flatten(1)
        x = self.mlp(x)
        center_norm = self.fc_center(x)
        normal_raw = self.fc_normal(x)
        cls_logits = self.fc_cls(x)
        return center_norm, normal_raw, cls_logits


class PoseROINet(torch.nn.Module):
    """
    forward(images, boxes_xyxy_px_list)
    train:  {"center_norm":[B,3], "normal":[B,3], "cls_logits":[B,nc]}
    eval :  {"center_m":[B,3],    "normal":[B,3], "cls_prob":[B,nc]}
    """
    def __init__(
        self,
        width,
        depth,
        csp,
        num_classes: int,
        img_size: int = 640,
        roi_out_size: int = 7,
        roi_sampling_ratio: int = 2,
        roi_ch: int = 256,
        mlp_hidden: int = 512,
        dropout: float = 0.0,

        # 新增：backbone 选择
        backbone: str = "darknet",  # "darknet" or "vit"
        vit_patch: int = 8,
        vit_dim: int = 512,
        vit_depth: int = 6,
        vit_heads: int = 8,
        vit_mlp_dim: int = 1024,
        vit_dim_head: int = 64,
        vit_dropout: float = 0.0,
        vit_emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.nc = int(num_classes)
        self.img_size = int(img_size)
        self.backbone = str(backbone)

        self.act = torch.nn.SiLU()

        self.fpn = DarkFPN(width, depth, csp)

        self.proj3 = torch.nn.Conv2d(width[3], roi_ch, kernel_size=1, bias=False)
        self.proj4 = torch.nn.Conv2d(width[4], roi_ch, kernel_size=1, bias=False)
        self.proj5 = torch.nn.Conv2d(width[5], roi_ch, kernel_size=1, bias=False)

        self.bn3 = torch.nn.BatchNorm2d(roi_ch, eps=0.001, momentum=0.03)
        self.bn4 = torch.nn.BatchNorm2d(roi_ch, eps=0.001, momentum=0.03)
        self.bn5 = torch.nn.BatchNorm2d(roi_ch, eps=0.001, momentum=0.03)
        self.act = torch.nn.SiLU()

        if self.backbone == "darknet":
            self.net = DarkNet(width, depth, csp)
            self.vit = None
            self.vit_to_p3 = None
            self.vit_to_p4 = None
            self.vit_to_p5 = None

        elif self.backbone == "vit":
            self.net = None

            self.vit = ViTBackbone(
                image_size=self.img_size,
                patch_size=vit_patch,   # 建议 8
                dim=vit_dim,
                depth=vit_depth,
                heads=vit_heads,
                mlp_dim=vit_mlp_dim,
                dim_head=vit_dim_head,
                dropout=vit_dropout,
                emb_dropout=vit_emb_dropout,
            )

            # 关键：造出“和 DarkNet 输出同通道”的 (p3_in,p4_in,p5_in)，再喂给同一个 DarkFPN
            # p3_in: stride=patch (建议 patch=8)
            self.vit_to_p3 = Conv(vit_dim,  width[3], torch.nn.SiLU(), k=1, s=1, p=0)
            # p4_in: stride=16
            self.vit_to_p4 = Conv(width[3], width[4], torch.nn.SiLU(), k=3, s=2, p=1)
            # p5_in: stride=32
            self.vit_to_p5 = Conv(width[4], width[5], torch.nn.SiLU(), k=3, s=2, p=1)

        else:
            raise ValueError(f"Unknown backbone='{self.backbone}', expected 'darknet' or 'vit'")

        # ROIAlign
        self.pooler = MultiScaleRoIAlign(
            featmap_names=["p3", "p4", "p5"],
            output_size=roi_out_size,
            sampling_ratio=roi_sampling_ratio,
        )

        # head 不变
        self.head = PoseROIHead(roi_ch, self.nc, hidden=mlp_hidden, dropout=dropout)

        # center 标准化统计量（米单位的 mean/std）
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

    def _fpn_feats(self, x: torch.Tensor):
        if self.backbone == "darknet":
            p3, p4, p5 = self.fpn(self.net(x))
        else:
            feat = self.vit(x)                 # [B, vit_dim, H/patch, W/patch] (建议 patch=8 -> 80x80)
            p3_in = self.vit_to_p3(feat)       # [B, width[3], 80, 80]
            p4_in = self.vit_to_p4(p3_in)      # [B, width[4], 40, 40]
            p5_in = self.vit_to_p5(p4_in)      # [B, width[5], 20, 20]
            p3, p4, p5 = self.fpn((p3_in, p4_in, p5_in))  # 同一个 DarkFPN

        # 两条路径都走同一套 proj+bn+act -> roi_ch
        p3 = self.act(self.bn3(self.proj3(p3)))
        p4 = self.act(self.bn4(self.proj4(p4)))
        p5 = self.act(self.bn5(self.proj5(p5)))
        return {"p3": p3, "p4": p4, "p5": p5}

    def forward(self, images: torch.Tensor, boxes: list):
        B, _, H, W = images.shape
        feats = self._fpn_feats(images)
        image_shapes = [(H, W)] * B

        roi_feat = self.pooler(feats, boxes, image_shapes)  # [sumK, C, ph, pw]

        # 兜底：每张图多框时取第一个框
        if roi_feat.shape[0] != B:
            out = []
            start = 0
            for b in range(B):
                k = int(boxes[b].shape[0])
                out.append(roi_feat[start:start+1])
                start += k
            roi_feat = torch.cat(out, dim=0)

        center_norm, normal_raw, cls_logits = self.head(roi_feat)

        # normal：直接单位化（你当前做法）
        normal = F.normalize(normal_raw, dim=1, eps=1e-6)

        if self.training:
            return {"center_norm": center_norm, "normal": normal, "cls_logits": cls_logits}

        if self.denorm_inference:
            center_m = center_norm * self.center_std + self.center_mean
        else:
            center_m = center_norm
        cls_prob = F.softmax(cls_logits, dim=1)
        return {"center_m": center_m, "normal": normal, "cls_prob": cls_prob, "cls_logits": cls_logits}

    def fuse(self):
        # 只 fuse Conv+BN（ViT 里 Linear 不处理）
        for m in self.modules():
            if type(m) is Conv and hasattr(m, "norm"):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, "norm")
        return self


def roi_pose(
    num_classes: int,
    img_size: int = 640,
    roi_ch: int = 256,
    backbone: str = "darknet",
    vit_patch: int = 8,
    vit_dim: int = 512,
    vit_depth: int = 6,
    vit_heads: int = 8,
    vit_mlp_dim: int = 1024,
):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]

    return PoseROINet(
        width, depth, csp, num_classes,
        img_size=img_size,
        roi_ch=roi_ch,
        backbone=backbone,
        vit_patch=vit_patch,
        vit_dim=vit_dim,
        vit_depth=vit_depth,
        vit_heads=vit_heads,
        vit_mlp_dim=vit_mlp_dim,
    )