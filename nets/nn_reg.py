import math
import os 
import torch
from torch import nn




def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, e=0.5):
        super().__init__()
        self.conv1 = Conv(ch, int(ch * e), torch.nn.SiLU(), k=3, p=1)
        self.conv2 = Conv(int(ch * e), ch, torch.nn.SiLU(), k=3, p=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CSPModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, torch.nn.SiLU())
        self.conv2 = Conv(in_ch, out_ch // 2, torch.nn.SiLU())
        self.conv3 = Conv(2 * (out_ch // 2), out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e=1.0),
                                         Residual(out_ch // 2, e=1.0))

    def forward(self, x):
        y = self.res_m(self.conv1(x))
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r):
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r), torch.nn.SiLU())
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch, torch.nn.SiLU())

        if not csp:
            self.res_m = torch.nn.ModuleList(Residual(out_ch // r) for _ in range(n))
        else:
            self.res_m = torch.nn.ModuleList(CSPModule(out_ch // r, out_ch // r) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2, torch.nn.SiLU())
        self.conv2 = Conv(in_ch * 2, out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat(tensors=[x, y1, y2, self.res_m(y2)], dim=1))


class Attention(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.num_head = num_head
        self.dim_head = ch // num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5

        self.qkv = Conv(ch, ch + self.dim_key * num_head * 2, torch.nn.Identity())

        self.conv1 = Conv(ch, ch, torch.nn.Identity(), k=3, p=1, g=ch)
        self.conv2 = Conv(ch, ch, torch.nn.Identity())

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)

        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
        return self.conv2(x)


class PSABlock(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.conv1 = Attention(ch, num_head)
        self.conv2 = torch.nn.Sequential(Conv(ch, ch * 2, torch.nn.SiLU()),
                                         Conv(ch * 2, ch, torch.nn.Identity()))

    def forward(self, x):
        x = x + self.conv1(x)
        return x + self.conv2(x)


class PSA(torch.nn.Module):
    def __init__(self, ch, n):
        super().__init__()
        self.conv1 = Conv(ch, 2 * (ch // 2), torch.nn.SiLU())
        self.conv2 = Conv(2 * (ch // 2), ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(*(PSABlock(ch // 2, ch // 128) for _ in range(n)))

    def forward(self, x):
        x, y = self.conv1(x).chunk(2, 1)
        return self.conv2(torch.cat(tensors=(x, self.res_m(y)), dim=1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], torch.nn.SiLU(), k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(width[1], width[2], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p2.append(CSP(width[2], width[3], depth[0], csp[0], r=4))
        # p3/8
        self.p3.append(Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p3.append(CSP(width[3], width[4], depth[1], csp[0], r=4))
        # p4/16
        self.p4.append(Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p4.append(CSP(width[4], width[4], depth[2], csp[1], r=2))
        # p5/32
        self.p5.append(Conv(width[4], width[5], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p5.append(CSP(width[5], width[5], depth[3], csp[1], r=2))
        self.p5.append(SPP(width[5], width[5]))
        self.p5.append(PSA(width[5], depth[4]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[5], csp[0], r=2)
        self.h2 = CSP(width[4] + width[4], width[3], depth[5], csp[0], r=2)
        self.h3 = Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h4 = CSP(width[3] + width[4], width[4], depth[5], csp[0], r=2)
        self.h5 = Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h6 = CSP(width[4] + width[5], width[5], depth[5], csp[1], r=2)

    def forward(self, x):
        p3, p4, p5 = x
        p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
        p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
        return p3, p4, p5


class DFL(torch.nn.Module):
    # Generalized Focal Loss
    # https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)




class AngleDistHead(nn.Module):
    """
    输入:  p5 特征 (B, C, H, W)（来自 DarkFPN 的第三层）
    输出:  dict:
      - angle_norm: (B, 3)  in [0,1]，分别对应你的 XYZ 分量化角度（后处理再映射到 [-180, 180] 等）
      - dist_log  : (B, 1)  距离的对数标度（训练时可用 L1/MSE，推理时对其 exp 取真值）
    """
    def __init__(self, in_channels, hidden=None):
        super().__init__()
        H = hidden if hidden is not None else max(128, in_channels // 2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.angle_head = nn.Sequential(
            nn.Conv2d(in_channels, H, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(H, 3, kernel_size=1, bias=True)   # -> [B,3,1,1]
        )
        self.dist_head = nn.Sequential(
            nn.Conv2d(in_channels, H, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(H, 1, kernel_size=1, bias=True)   # -> [B,1,1,1]
        )

        # 初始化：角度最后一层偏置置零；距离同理（更容易稳定）
        with torch.no_grad():
            for m in (self.angle_head, self.dist_head):
                if isinstance(m[-1], nn.Conv2d):
                    m[-1].bias.zero_()

    def forward(self, p5):
        g = self.pool(p5)                    # [B, C5, 1, 1]
        angle = torch.sigmoid(self.angle_head(g)).flatten(1)  # [B,3], 0..1
        dist  = self.dist_head(g).flatten(1)                  # [B,1] (log 距离)
        return {"angle_norm": angle, "dist_log": dist}




class YOLOAngleDist(nn.Module):
    """
    DarkNet + DarkFPN 作为骨干与颈部；只输出 angle_norm / dist_log 两个分支。
    与 YOLOMultiTask/Head 没有依赖关系，不使用 anchors/DFL/cls。
    """
    def __init__(self, width, depth, csp, head_hidden: int = 512):
        super().__init__()
        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)

        # 使用 p5 做全局回归
        C_p5 = width[5]
        self.head = AngleDistHead(in_channels=C_p5, hidden=head_hidden)

        # 可选：推理或日志中若需要 stride，可计算一遍（但不再用于检测）
        with torch.no_grad():
            img_dummy = torch.zeros(1, width[0], 256, 256)
            p3, p4, p5 = self.fpn(self.net(img_dummy))
            self.stride = torch.tensor([256 / p.shape[-2] for p in (p3, p4, p5)], dtype=torch.float)

    def forward(self, x):
        p3, p4, p5 = self.fpn(self.net(x))
        return self.head(p5)

    def fuse(self):
        # 仍支持卷积+BN 融合（便于部署）
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def _torch_load_safely(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

# def _extract_state_dict(obj):
#     if isinstance(obj, dict):
#         if "state_dict" in obj and isinstance(obj["state_dict"], (dict,)):
#             sd = obj["state_dict"]
#         elif "model" in obj and isinstance(obj["model"], (dict,)):
#             sd = obj["model"]
#         else:
#             if all(isinstance(v, torch.Tensor) for v in obj.values()):
#                 sd = obj
#             else:
#                 raise ValueError("无法从 checkpoint dict 中确定 state_dict，请检查键名（期望 'state_dict' 或 'model'）。")
#     else:
#         if hasattr(obj, "state_dict"):
#             sd = obj.state_dict()
#         else:
#             raise ValueError(f"未知的 checkpoint 类型：{type(obj)}，无法提取 state_dict。")
#     return sd
def _extract_state_dict(obj):
    """
    从多种常见 checkpoint 结构中提取 state_dict。
    兼容：纯 state_dict、{'state_dict': ...}、{'model': ...}、{'ema': module/...}、以及常见别名键。
    """
    import torch
    from collections import OrderedDict

    # 如果是“整个模型/EMA 模型对象”
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")) and not isinstance(obj, (dict,)):
        return obj.state_dict()

    if isinstance(obj, (dict, OrderedDict)):
        # 1) 首选常见键
        candidate_keys = [
            "state_dict", "model", "ema", "model_ema",
            "ema_state_dict", "model_state",
            "net", "network", "weights", "params",  # 兼容各种训练脚本
        ]
        for k in candidate_keys:
            if k in obj:
                v = obj[k]
                # a) 直接是 dict 且值基本都是 Tensor
                if isinstance(v, (dict, OrderedDict)) and all(isinstance(x, torch.Tensor) for x in v.values()):
                    return v
                # b) 是模块对象
                if hasattr(v, "state_dict") and callable(getattr(v, "state_dict")):
                    return v.state_dict()

        # 2) 直接就是纯 state_dict？
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj

        # 3) 再做一层浅层扫描：某个 value 本身是“看起来像 state_dict”的 dict
        for k, v in obj.items():
            if isinstance(v, (dict, OrderedDict)) and v:
                if all(isinstance(x, torch.Tensor) for x in v.values()):
                    return v
                # 某些脚本把 state_dict 再包了一层 model/ema 对象字段
                if hasattr(v, "state_dict") and callable(getattr(v, "state_dict")):
                    return v.state_dict()

        # 4) 实在不行，给个提示
        raise ValueError(
            "无法从 checkpoint dict 中确定 state_dict，可用键包括："
            + ", ".join(map(str, obj.keys()))
            + "。请检查权重文件的结构。"
        )

    # 其它类型：尝试当作模块
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        return obj.state_dict()

    raise ValueError(f"未知的 checkpoint 类型：{type(obj)}，无法提取 state_dict。")

def _strip_prefix(sd, prefixes=("module.", "model.")):
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd

def _filter_heads_for_pretrain(sd):
    """
    预训练阶段通常不加载任务头（图像级 + 检测头）。
    """
    drop_prefixes = ("cls_head.", "angle_head.", "dist_head.", "det_head.")
    return {k: v for k, v in sd.items() if not any(k.startswith(p) for p in drop_prefixes)}

def _shape_compatible_only(model, sd):
    msd = model.state_dict()
    filtered, skipped = {}, []
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    return filtered, skipped

def _load_checkpoint_to_model(model, ckpt_path, strict=False, drop_heads=False, tag=""):
    obj = _torch_load_safely(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(obj)
    sd = _strip_prefix(sd)
    if drop_heads:
        sd = _filter_heads_for_pretrain(sd)
    sd, skipped = _shape_compatible_only(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=strict)

    print(f"[{tag}] Loaded from: {ckpt_path}")
    print(f"[{tag}] Loaded keys: {len(sd)} | Skipped (shape mismatch): {len(skipped)}")
    if missing:
        print(f"[{tag}] Missing keys ({len(missing)}): {sorted(missing)[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[{tag}] Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:8]}{' ...' if len(unexpected)>8 else ''}")



def build_model(args):
    device = torch.device(getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    # 解析骨干配置
    width, depth, csp = _resolve_yolo_cfg(args)

    # 超参
    head_hidden = int(getattr(args, "head_hidden", 512))

    # 构建仅回归模型
    model = YOLOAngleDist(width=width, depth=depth, csp=csp, head_hidden=head_hidden).to(device)

    mode = str(getattr(args, "mode", "train")).lower()
    pretrain = bool(getattr(args, "pretrain", False))

    if mode in ("train", "training"):
        if pretrain:
            pretrain_path = getattr(args, "pretrain_path", None)
            if pretrain_path is None or not os.path.isfile(pretrain_path):
                print("[Pretrain] 未提供有效的 args.pretrain_path，跳过加载预训练权重。")
            else:
                # 预训练：通常只加载 backbone+fpn，丢弃旧的检测/分类头
                _load_checkpoint_to_model(
                    model, pretrain_path,
                    strict=False, drop_heads=True, tag="Pretrain(Backbone+FPN only)"
                )
        model.train()

    elif mode in ("test", "eval", "evaluation", "inference"):
        ckpt = getattr(args, "ckpt", None) or getattr(args, "checkpoint", None) or getattr(args, "weights", None)
        if ckpt is None or not os.path.isfile(ckpt):
            print("[Eval] 未提供有效权重路径，将以随机初始化评估。")
        else:
            _load_checkpoint_to_model(
                model, ckpt, strict=False, drop_heads=False, tag="Eval(Full)"
            )
        model.eval()

    else:
        print(f"[Warn] 未识别的 args.mode='{getattr(args, 'mode', None)}'，默认按训练模式处理。")
        model.train()

    return model



def _resolve_yolo_cfg(args):
    """
    支持三种方式指定骨干宽度/深度：
      1) 直接给 args.width / args.depth / args.csp
      2) 指定 args.size 或 args.variant in {n,t,s,m,l,x}
      3) 默认 's'
    返回 (width, depth, csp)
    """
    if hasattr(args, "width") and hasattr(args, "depth") and hasattr(args, "csp") and args.width and args.depth and args.csp:
        return args.width, args.depth, args.csp

    size = (getattr(args, "size", None) or getattr(args, "variant", "s")).lower()
    if size == "n":
        csp   = [False, True]
        depth = [1, 1, 1, 1, 1, 1]
        width = [3, 16, 32, 64, 128, 256]
    elif size == "t":
        csp   = [False, True]
        depth = [1, 1, 1, 1, 1, 1]
        width = [3, 24, 48, 96, 192, 384]
    elif size == "s":
        csp   = [False, True]
        depth = [1, 1, 1, 1, 1, 1]
        width = [3, 32, 64, 128, 256, 512]
    elif size == "m":
        csp   = [True, True]
        depth = [1, 1, 1, 1, 1, 1]
        width = [3, 64, 128, 256, 512, 512]
    elif size == "l":
        csp   = [True, True]
        depth = [2, 2, 2, 2, 2, 2]
        width = [3, 64, 128, 256, 512, 512]
    elif size == "x":
        csp   = [True, True]
        depth = [2, 2, 2, 2, 2, 2]
        width = [3, 96, 192, 384, 768, 768]
    else:
        raise ValueError(f"Unknown YOLO size/variant: {size}")
    return width, depth, csp


def yolo_v11_n():
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 16, 32, 64, 128, 256]
    return YOLOAngleDist(width, depth, csp)

def yolo_v11_t():
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 24, 48, 96, 192, 384]
    return YOLOAngleDist(width, depth, csp)

def yolo_v11_s():
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 32, 64, 128, 256, 512]
    return YOLOAngleDist(width, depth, csp)

def yolo_v11_m():
    csp = [True, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 64, 128, 256, 512, 512]
    return YOLOAngleDist(width, depth, csp)

def yolo_v11_l():
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 64, 128, 256, 512, 512]
    return YOLOAngleDist(width, depth, csp)

def yolo_v11_x():
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    return YOLOAngleDist(width, depth, csp)
















