# nets/nn_roi_pose.py
import torch
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

# 复用现有 backbone/fpn/Conv/fuse_conv
from nets.nn_center_and_normal import DarkNet, DarkFPN, Conv, fuse_conv


class PoseROIHead(torch.nn.Module):
    """
    输入:
      roi_feat_local : [B, C, ph, pw]
      roi_feat_global: [B, C, ph, pw] (可选)
    融合:
      pool->vec 后 concat，再 MLP 回归
    """
    def __init__(
        self,
        in_ch: int,
        nc: int,
        hidden: int = 512,
        dropout: float = 0.0,
        use_global: bool = True,
    ):
        super().__init__()
        self.nc = int(nc)
        self.use_global = bool(use_global)

        # local 分支
        self.local_conv = torch.nn.Sequential(
            Conv(in_ch, in_ch, torch.nn.SiLU(), k=3, p=1),
            Conv(in_ch, in_ch, torch.nn.SiLU(), k=3, p=1),
        )

        # global 分支（可选）
        if self.use_global:
            self.global_conv = torch.nn.Sequential(
                Conv(in_ch, in_ch, torch.nn.SiLU(), k=3, p=1),
                Conv(in_ch, in_ch, torch.nn.SiLU(), k=3, p=1),
            )
            fuse_in = in_ch * 2
        else:
            self.global_conv = None
            fuse_in = in_ch

        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        mlp = [
            torch.nn.Linear(fuse_in, hidden),
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

    def forward(self, roi_feat_local, roi_feat_global):
        xl = self.local_conv(roi_feat_local)
        xl = self.pool(xl).flatten(1)  # [B,C]

        if self.use_global:
            assert roi_feat_global is not None, "use_global=True but roi_feat_global is None"
            xg = self.global_conv(roi_feat_global)
            xg = self.pool(xg).flatten(1)  # [B,C]
            x = torch.cat([xl, xg], dim=1)  # [B,2C]
        else:
            x = xl  # [B,C]

        x = self.mlp(x)
        center_norm = self.fc_center(x)
        normal_raw = self.fc_normal(x)
        cls_logits = self.fc_cls(x)
        return center_norm, normal_raw, cls_logits


class PoseROINet(torch.nn.Module):
    """
    forward(images, boxes_local, boxes_global)

    boxes_local : list length B, each [K,4] xyxy pixel coords  (建议K=1)
    boxes_global: list length B, each [K,4] xyxy pixel coords  (建议K=1)

    train:  {"center_norm":[B,3], "normal":[B,3], "cls_logits":[B,nc]}
    eval :  {"center_m":[B,3],    "normal":[B,3], "cls_prob":[B,nc], "cls_logits":[B,nc]}
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
    ):
        super().__init__()
        self.nc = int(num_classes)
        self.img_size = int(img_size)
        self.use_global = bool(use_global)

        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)

        # 多尺度 ROIAlign 要求各层通道一致，所以投影到 roi_ch
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

        self.head = PoseROIHead(
            roi_ch, self.nc,
            hidden=mlp_hidden,
            dropout=dropout,
            use_global=self.use_global
        )

        # center 标准化统计量（米单位 mean/std）
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

    @staticmethod
    def _take_first_box_per_image(roi_feat: torch.Tensor, boxes: list) -> torch.Tensor:
        # roi_feat: [sumK, C, ph, pw]  ->  [B, C, ph, pw] by taking first per image
        # boxes[b] could have K>1
        B = len(boxes)
        if roi_feat.shape[0] == B:
            return roi_feat
        out = []
        start = 0
        for b in range(B):
            k = int(boxes[b].shape[0])
            out.append(roi_feat[start:start + 1])
            start += k
        return torch.cat(out, dim=0)

    def forward(self, images, boxes_local, boxes_global):
        B, _, H, W = images.shape
        feats = self._fpn_feats(images)
        image_shapes = [(H, W)] * B

        if boxes_global is None:
            boxes_global = boxes_local

        roi_local = self.pooler(feats, boxes_local, image_shapes)    # [sumK, C, ph, pw]
        roi_local = self._take_first_box_per_image(roi_local, boxes_local)

        if self.use_global:
            roi_global = self.pooler(feats, boxes_global, image_shapes)
            roi_global = self._take_first_box_per_image(roi_global, boxes_global)
        else:
            roi_global = None

        center_norm, normal_raw, cls_logits = self.head(roi_local, roi_global)

        # normal：不 tanh，只单位化（你现在用的方案）
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


def roi_pose_v11_x(num_classes: int, img_size: int = 640, roi_ch: int = 256, use_global: bool = True):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    return PoseROINet(width, depth, csp, num_classes, img_size=img_size, roi_ch=roi_ch, use_global=use_global)