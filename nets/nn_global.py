# nets/nn_roi_pose.py  (GLOBAL-ONLY version)
import torch
import torch.nn.functional as F


from nets.nn_center_and_normal import DarkNet, DarkFPN, Conv, fuse_conv


class PoseGlobalHead(torch.nn.Module):
    def __init__(self, in_dim: int, nc: int, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.nc = int(nc)

        mlp = [
            torch.nn.Linear(in_dim, hidden),
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

    def forward(self, global_vec: torch.Tensor):
        x = self.mlp(global_vec)
        center_norm = self.fc_center(x)
        normal_raw = self.fc_normal(x)
        cls_logits = self.fc_cls(x)
        return center_norm, normal_raw, cls_logits


class PoseGlobalNet(torch.nn.Module):
    """
    Global-only Pose Net (no boxes, no ROIAlign, no detector)

    forward(images)
    train: {"center_norm":[B,3], "normal":[B,3], "cls_logits":[B,nc]}
    eval : {"center_m":[B,3],    "normal":[B,3], "cls_prob":[B,nc], "cls_logits":[B,nc]}
    """
    def __init__(
        self,
        width,
        depth,
        csp,
        num_classes: int,
        img_size: int = 640,
        roi_ch: int = 256,
        mlp_hidden: int = 512,
        dropout: float = 0.0,
        global_from: str = "p5",   # "p5" or "p345"
    ):
        super().__init__()
        self.nc = int(num_classes)
        self.img_size = int(img_size)
        self.global_from = str(global_from)

        assert self.global_from in ("p5", "p345"), "global_from must be 'p5' or 'p345'"

        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)

        self.proj3 = torch.nn.Conv2d(width[3], roi_ch, kernel_size=1, bias=False)
        self.proj4 = torch.nn.Conv2d(width[4], roi_ch, kernel_size=1, bias=False)
        self.proj5 = torch.nn.Conv2d(width[5], roi_ch, kernel_size=1, bias=False)

        self.bn3 = torch.nn.BatchNorm2d(roi_ch, eps=0.001, momentum=0.03)
        self.bn4 = torch.nn.BatchNorm2d(roi_ch, eps=0.001, momentum=0.03)
        self.bn5 = torch.nn.BatchNorm2d(roi_ch, eps=0.001, momentum=0.03)
        self.act = torch.nn.SiLU()

        self.global_conv = torch.nn.Sequential(
            Conv(roi_ch, roi_ch, torch.nn.SiLU(), k=3, p=1),
            Conv(roi_ch, roi_ch, torch.nn.SiLU(), k=3, p=1),
        )
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))


        in_dim = roi_ch if self.global_from == "p5" else roi_ch * 3
        self.head = PoseGlobalHead(in_dim, self.nc, hidden=mlp_hidden, dropout=dropout)

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
        p3, p4, p5 = self.fpn(self.net(x))
        p3 = self.act(self.bn3(self.proj3(p3)))
        p4 = self.act(self.bn4(self.proj4(p4)))
        p5 = self.act(self.bn5(self.proj5(p5)))
        return p3, p4, p5  # all are roi_ch channels now

    def _global_vec(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> torch.Tensor:
        def encode(feat: torch.Tensor) -> torch.Tensor:
            feat = self.global_conv(feat)
            return self.gap(feat).flatten(1)  # [B, roi_ch]

        if self.global_from == "p5":
            return encode(p5)  # [B, roi_ch]

        # p345: concat [B, roi_ch*3]
        v3 = encode(p3)
        v4 = encode(p4)
        v5 = encode(p5)
        return torch.cat([v3, v4, v5], dim=1)

    def forward(self, images: torch.Tensor):
        """
        images: [B,3,H,W]
        """
        p3, p4, p5 = self._fpn_feats(images)
        g = self._global_vec(p3, p4, p5)  # [B, D]

        center_norm, normal_raw, cls_logits = self.head(g)

        # normal
        normal = F.normalize(normal_raw, dim=1, eps=1e-6)

        if self.training:
            return {"center_norm": center_norm, "normal": normal, "cls_logits": cls_logits}

        # eval
        if self.denorm_inference:
            center_m = center_norm * self.center_std + self.center_mean
        else:
            center_m = center_norm

        cls_prob = F.softmax(cls_logits, dim=1)
        return {"center_m": center_m, "normal": normal, "cls_prob": cls_prob, "cls_logits": cls_logits}

    def fuse(self):
        # fuse Conv+BN
        for m in self.modules():
            if type(m) is Conv and hasattr(m, "norm"):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, "norm")
        return self


def global_pose_v11_x(
    num_classes: int,
    img_size: int = 640,
    roi_ch: int = 256,
    global_from: str = "p5",  # "p5" or "p345"
):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    return PoseGlobalNet(
        width, depth, csp,
        num_classes=num_classes,
        img_size=img_size,
        roi_ch=roi_ch,
        global_from=global_from,
    )