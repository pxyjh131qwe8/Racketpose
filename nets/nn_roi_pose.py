

import torch
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from nets.nn_center_and_normal import DarkNet, DarkFPN, Conv, fuse_conv


class PoseROIHead(torch.nn.Module):
    def __init__(self, in_ch: int, nc: int, hidden: int = 512, dropout: float = 0.0, use_global: bool = True):
        super().__init__()
        self.nc = int(nc)
        self.in_ch = int(in_ch)
        self.use_global = bool(use_global)

        self.conv = torch.nn.Sequential(
            Conv(in_ch, in_ch, torch.nn.SiLU(), k=3, p=1),
            Conv(in_ch, in_ch, torch.nn.SiLU(), k=3, p=1),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        mlp_in = in_ch * (2 if self.use_global else 1)

        mlp = [
            torch.nn.Linear(mlp_in, hidden),
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

    def forward(self, roi_feat, global_vec):
        """
        roi_feat:   [B, C, ph, pw]
        global_vec: [B, C] or None
        """
        x = self.conv(roi_feat)
        roi_vec = self.pool(x).flatten(1)  # [B,C]

        if self.use_global:
            assert global_vec is not None, "use_global=True but global_vec is None"
            feat = torch.cat([roi_vec, global_vec], dim=1)  # [B,2C]
        else:
            feat = roi_vec

        feat = self.mlp(feat)
        center_norm = self.fc_center(feat)
        normal_raw = self.fc_normal(feat)
        cls_logits = self.fc_cls(feat)
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
        use_global: bool = True,     
        global_from: str = "p5",      
    ):
        super().__init__()
        self.nc = int(num_classes)
        self.img_size = int(img_size)
        self.use_global = bool(use_global)
        self.global_from = str(global_from)

        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)

        self.proj3 = torch.nn.Conv2d(width[3], roi_ch, kernel_size=1, bias=False)
        self.proj4 = torch.nn.Conv2d(width[4], roi_ch, kernel_size=1, bias=False)
        self.proj5 = torch.nn.Conv2d(width[5], roi_ch, kernel_size=1, bias=False)

        self.bn3 = torch.nn.BatchNorm2d(roi_ch, eps=0.001, momentum=0.03)
        self.bn4 = torch.nn.BatchNorm2d(roi_ch, eps=0.001, momentum=0.03)
        self.bn5 = torch.nn.BatchNorm2d(roi_ch, eps=0.001, momentum=0.03)
        self.act = torch.nn.SiLU()

        self.pooler = MultiScaleRoIAlign(
            featmap_names=["p3", "p4", "p5"],
            output_size=roi_out_size,
            sampling_ratio=roi_sampling_ratio,
        )

        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.head = PoseROIHead(
            roi_ch, self.nc, hidden=mlp_hidden, dropout=dropout, use_global=self.use_global
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

    def _fpn_feats(self, x: torch.Tensor):
        p3, p4, p5 = self.fpn(self.net(x))
        p3 = self.act(self.bn3(self.proj3(p3)))
        p4 = self.act(self.bn4(self.proj4(p4)))
        p5 = self.act(self.bn5(self.proj5(p5)))
        return {"p3": p3, "p4": p4, "p5": p5}

    def _global_vec(self, feats: dict) -> torch.Tensor:
        
        if not self.use_global:
            return None

        if self.global_from == "p5":
            # print("Using global vector from p5 only")
            g = self.gap(feats["p5"]).flatten(1)  # [B,C]
            return g

        if self.global_from == "p345":
            g3 = self.gap(feats["p3"]).flatten(1)
            g4 = self.gap(feats["p4"]).flatten(1)
            g5 = self.gap(feats["p5"]).flatten(1)
            return (g3 + g4 + g5) / 3.0

        raise ValueError(f"Unknown global_from={self.global_from}, expected 'p5' or 'p345'")

    def forward(self, images: torch.Tensor, boxes: list):
        """
        images: [B,3,H,W]
        boxes : list length B, each tensor [K,4] in xyxy pixel coords
        """
        B, _, H, W = images.shape
        feats = self._fpn_feats(images)
        image_shapes = [(H, W)] * B

        global_vec = self._global_vec(feats) if self.use_global else None

        roi_feat = self.pooler(feats, boxes, image_shapes)  # [sumK, C, ph, pw]

        if roi_feat.shape[0] != B:
            out = []
            start = 0
            for b in range(B):
                k = int(boxes[b].shape[0])
                out.append(roi_feat[start:start+1])
                start += k
            roi_feat = torch.cat(out, dim=0)

        center_norm, normal_raw, cls_logits = self.head(roi_feat, global_vec=global_vec)

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
        for m in self.modules():
            if type(m) is Conv and hasattr(m, "norm"):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, "norm")
        return self


def roi_pose_v11_x(
    num_classes: int,
    img_size: int = 640,
    roi_ch: int = 256,
    use_global: bool = True,
    global_from: str = "p5",
):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    return PoseROINet(
        width, depth, csp, num_classes,
        img_size=img_size,
        roi_ch=roi_ch,
        use_global=use_global,
        global_from=global_from,
    )