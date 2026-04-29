# import math
# import os 
# import torch
# from torch import nn

# from utils.util import make_anchors


# def fuse_conv(conv, norm):
#     fused_conv = torch.nn.Conv2d(conv.in_channels,
#                                  conv.out_channels,
#                                  kernel_size=conv.kernel_size,
#                                  stride=conv.stride,
#                                  padding=conv.padding,
#                                  groups=conv.groups,
#                                  bias=True).requires_grad_(False).to(conv.weight.device)

#     w_conv = conv.weight.clone().view(conv.out_channels, -1)
#     w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
#     fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

#     b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
#     b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
#     fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

#     return fused_conv


# class Conv(torch.nn.Module):
#     def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, g=1):
#         super().__init__()
#         self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
#         self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
#         self.relu = activation

#     def forward(self, x):
#         return self.relu(self.norm(self.conv(x)))

#     def fuse_forward(self, x):
#         return self.relu(self.conv(x))


# class Residual(torch.nn.Module):
#     def __init__(self, ch, e=0.5):
#         super().__init__()
#         self.conv1 = Conv(ch, int(ch * e), torch.nn.SiLU(), k=3, p=1)
#         self.conv2 = Conv(int(ch * e), ch, torch.nn.SiLU(), k=3, p=1)

#     def forward(self, x):
#         return x + self.conv2(self.conv1(x))


# class CSPModule(torch.nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv1 = Conv(in_ch, out_ch // 2, torch.nn.SiLU())
#         self.conv2 = Conv(in_ch, out_ch // 2, torch.nn.SiLU())
#         self.conv3 = Conv(2 * (out_ch // 2), out_ch, torch.nn.SiLU())
#         self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e=1.0),
#                                          Residual(out_ch // 2, e=1.0))

#     def forward(self, x):
#         y = self.res_m(self.conv1(x))
#         return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


# class CSP(torch.nn.Module):
#     def __init__(self, in_ch, out_ch, n, csp, r):
#         super().__init__()
#         self.conv1 = Conv(in_ch, 2 * (out_ch // r), torch.nn.SiLU())
#         self.conv2 = Conv((2 + n) * (out_ch // r), out_ch, torch.nn.SiLU())

#         if not csp:
#             self.res_m = torch.nn.ModuleList(Residual(out_ch // r) for _ in range(n))
#         else:
#             self.res_m = torch.nn.ModuleList(CSPModule(out_ch // r, out_ch // r) for _ in range(n))

#     def forward(self, x):
#         y = list(self.conv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.res_m)
#         return self.conv2(torch.cat(y, dim=1))


# class SPP(torch.nn.Module):
#     def __init__(self, in_ch, out_ch, k=5):
#         super().__init__()
#         self.conv1 = Conv(in_ch, in_ch // 2, torch.nn.SiLU())
#         self.conv2 = Conv(in_ch * 2, out_ch, torch.nn.SiLU())
#         self.res_m = torch.nn.MaxPool2d(k, stride=1, padding=k // 2)

#     def forward(self, x):
#         x = self.conv1(x)
#         y1 = self.res_m(x)
#         y2 = self.res_m(y1)
#         return self.conv2(torch.cat(tensors=[x, y1, y2, self.res_m(y2)], dim=1))


# class Attention(torch.nn.Module):

#     def __init__(self, ch, num_head):
#         super().__init__()
#         self.num_head = num_head
#         self.dim_head = ch // num_head
#         self.dim_key = self.dim_head // 2
#         self.scale = self.dim_key ** -0.5

#         self.qkv = Conv(ch, ch + self.dim_key * num_head * 2, torch.nn.Identity())

#         self.conv1 = Conv(ch, ch, torch.nn.Identity(), k=3, p=1, g=ch)
#         self.conv2 = Conv(ch, ch, torch.nn.Identity())

#     def forward(self, x):
#         b, c, h, w = x.shape

#         qkv = self.qkv(x)
#         qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)

#         q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

#         attn = (q.transpose(-2, -1) @ k) * self.scale
#         attn = attn.softmax(dim=-1)

#         x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
#         return self.conv2(x)


# class PSABlock(torch.nn.Module):

#     def __init__(self, ch, num_head):
#         super().__init__()
#         self.conv1 = Attention(ch, num_head)
#         self.conv2 = torch.nn.Sequential(Conv(ch, ch * 2, torch.nn.SiLU()),
#                                          Conv(ch * 2, ch, torch.nn.Identity()))

#     def forward(self, x):
#         x = x + self.conv1(x)
#         return x + self.conv2(x)


# class PSA(torch.nn.Module):
#     def __init__(self, ch, n):
#         super().__init__()
#         self.conv1 = Conv(ch, 2 * (ch // 2), torch.nn.SiLU())
#         self.conv2 = Conv(2 * (ch // 2), ch, torch.nn.SiLU())
#         self.res_m = torch.nn.Sequential(*(PSABlock(ch // 2, ch // 128) for _ in range(n)))

#     def forward(self, x):
#         x, y = self.conv1(x).chunk(2, 1)
#         return self.conv2(torch.cat(tensors=(x, self.res_m(y)), dim=1))


# class DarkNet(torch.nn.Module):
#     def __init__(self, width, depth, csp):
#         super().__init__()
#         self.p1 = []
#         self.p2 = []
#         self.p3 = []
#         self.p4 = []
#         self.p5 = []

#         # p1/2
#         self.p1.append(Conv(width[0], width[1], torch.nn.SiLU(), k=3, s=2, p=1))
#         # p2/4
#         self.p2.append(Conv(width[1], width[2], torch.nn.SiLU(), k=3, s=2, p=1))
#         self.p2.append(CSP(width[2], width[3], depth[0], csp[0], r=4))
#         # p3/8
#         self.p3.append(Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1))
#         self.p3.append(CSP(width[3], width[4], depth[1], csp[0], r=4))
#         # p4/16
#         self.p4.append(Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1))
#         self.p4.append(CSP(width[4], width[4], depth[2], csp[1], r=2))
#         # p5/32
#         self.p5.append(Conv(width[4], width[5], torch.nn.SiLU(), k=3, s=2, p=1))
#         self.p5.append(CSP(width[5], width[5], depth[3], csp[1], r=2))
#         self.p5.append(SPP(width[5], width[5]))
#         self.p5.append(PSA(width[5], depth[4]))

#         self.p1 = torch.nn.Sequential(*self.p1)
#         self.p2 = torch.nn.Sequential(*self.p2)
#         self.p3 = torch.nn.Sequential(*self.p3)
#         self.p4 = torch.nn.Sequential(*self.p4)
#         self.p5 = torch.nn.Sequential(*self.p5)

#     def forward(self, x):
#         p1 = self.p1(x)
#         p2 = self.p2(p1)
#         p3 = self.p3(p2)
#         p4 = self.p4(p3)
#         p5 = self.p5(p4)
#         return p3, p4, p5


# class DarkFPN(torch.nn.Module):
#     def __init__(self, width, depth, csp):
#         super().__init__()
#         self.up = torch.nn.Upsample(scale_factor=2)
#         self.h1 = CSP(width[4] + width[5], width[4], depth[5], csp[0], r=2)
#         self.h2 = CSP(width[4] + width[4], width[3], depth[5], csp[0], r=2)
#         self.h3 = Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1)
#         self.h4 = CSP(width[3] + width[4], width[4], depth[5], csp[0], r=2)
#         self.h5 = Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1)
#         self.h6 = CSP(width[4] + width[5], width[5], depth[5], csp[1], r=2)

#     def forward(self, x):
#         p3, p4, p5 = x
#         p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
#         p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
#         p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
#         p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
#         return p3, p4, p5


# class DFL(torch.nn.Module):
#     # Generalized Focal Loss
#     # https://ieeexplore.ieee.org/document/9792391
#     def __init__(self, ch=16):
#         super().__init__()
#         self.ch = ch
#         self.conv = torch.nn.Conv2d(ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
#         x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
#         self.conv.weight.data[:] = torch.nn.Parameter(x)

#     def forward(self, x):
#         b, c, a = x.shape
#         x = x.view(b, 4, self.ch, a).transpose(2, 1)
#         return self.conv(x.softmax(1)).view(b, 4, a)


# class Head(torch.nn.Module):
#     anchors = torch.empty(0)
#     strides = torch.empty(0)

#     def __init__(self, nc=80, filters=()):
#         super().__init__()
#         self.ch = 16  # DFL channels
#         self.nc = nc  # number of classes
#         self.nl = len(filters)  # number of detection layers
#         self.no = nc + self.ch * 4  # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)  # strides computed during build

#         box = max(64, filters[0] // 4)
#         cls = max(80, filters[0], self.nc)

#         self.dfl = DFL(self.ch)
#         self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box,torch.nn.SiLU(), k=3, p=1),
#                                                            Conv(box, box,torch.nn.SiLU(), k=3, p=1),
#                                                            torch.nn.Conv2d(box, out_channels=4 * self.ch,
#                                                                            kernel_size=1)) for x in filters)
#         self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, x, torch.nn.SiLU(), k=3, p=1, g=x),
#                                                            Conv(x, cls, torch.nn.SiLU()),
#                                                            Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
#                                                            Conv(cls, cls, torch.nn.SiLU()),
#                                                            torch.nn.Conv2d(cls, out_channels=self.nc,
#                                                                            kernel_size=1)) for x in filters)

#     def forward(self, x):
#         for i, (box, cls) in enumerate(zip(self.box, self.cls)):
#             x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
#         if self.training:
#             return x

#         self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
#         x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
#         box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

#         a, b = self.dfl(box).chunk(2, 1)
#         a = self.anchors.unsqueeze(0) - a
#         b = self.anchors.unsqueeze(0) + b
#         box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

#         return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

#     def initialize_biases(self):
#         # Initialize biases
#         # WARNING: requires stride availability
#         for box, cls, s in zip(self.box, self.cls, self.stride):
#             # box
#             box[-1].bias.data[:] = 1.0
#             # cls (.01 objects, 80 classes, 640 image)
#             cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


# class MultiTaskHead(torch.nn.Module):
#     anchors = torch.empty(0) 
#     strides = torch.empty(0) 
    
#     def __init__(self, nc=4, filters=()):
#         super().__init__() 
#         self.ch = 16                 # DFL channels
#         self.nc = nc                 # number of classes
#         self.nl = len(filters)       # number of detection layers
#         self.no = nc + self.ch * 4   # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)

#         box = max(64, filters[0] // 4)
#         cls = max(80, filters[0], self.nc)
        
#         self.dfl = DFL(self.ch)
#         self.box = torch.nn.ModuleList(
#             torch.nn.Sequential(
#                 Conv(x, box, torch.nn.SiLU(), k=3, p=1),
#                 Conv(box, box, torch.nn.SiLU(), k=3, p=1),
#                 torch.nn.Conv2d(box, out_channels=4 * self.ch, kernel_size=1)
#             ) for x in filters
#         )
#         self.cls = torch.nn.ModuleList(
#             torch.nn.Sequential(
#                 Conv(x, x,   torch.nn.SiLU(), k=3, p=1, g=x),
#                 Conv(x, cls, torch.nn.SiLU()),
#                 Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
#                 Conv(cls, cls, torch.nn.SiLU()),
#                 torch.nn.Conv2d(cls, out_channels=self.nc, kernel_size=1)
#             ) for x in filters
#         )
        
#         # 新增基于p5的全局回归头（角度/距离） 
#         C_p5 = filters[-1]                  # 最后一层(p5)的通道
#         H = max(128, C_p5 // 2)            # 隐层宽度，经验值
#         self.pool = torch.nn.AdaptiveAvgPool2d(1)
#         self.angle_head = torch.nn.Sequential(
#             torch.nn.Conv2d(C_p5, H, kernel_size=1, bias=True),
#             torch.nn.SiLU(),
#             torch.nn.Conv2d(H, 3, kernel_size=1, bias=True)  # -> [B,3,1,1]
#         )
#         self.dist_head  = torch.nn.Sequential(
#             torch.nn.Conv2d(C_p5, H, kernel_size=1, bias=True),
#             torch.nn.SiLU(),
#             torch.nn.Conv2d(H, 1, kernel_size=1, bias=True)  # -> [B,1,1,1]
#         )

#         # 轻微初始化（
#         with torch.no_grad():
#             for m in (self.angle_head, self.dist_head):
#                 if isinstance(m[-1], torch.nn.Conv2d):
#                     m[-1].bias.zero_()
        
#     def forward(self, x, return_aux: bool = True):
#         """
#         x: list[p3, p4, p5]，与原 YOLO 一致
#         return_aux=False: 完全保持原返回
#         return_aux=True : 额外返回 {"angle_norm": [B,3], "dist_log": [B,1]}
#         """  
#         # 保存一份原始三层特征
#         feats = [xi for xi in x] 
        
#         # 检测头前向 
#         for i, (box, cls) in enumerate(zip(self.box, self.cls)):
#             x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        
#         # 训练直接返回list，与原实现一致
#         if self.training:
#             if return_aux:
#                 # p5 做全局回归
#                 p5 = feats[-1]
#                 g  = self.pool(p5)
#                 angle = torch.sigmoid(self.angle_head(g)).flatten(1)  # [B,3]
#                 dist  = self.dist_head(g).flatten(1)                  # [B,1]
#                 aux = {"angle_norm": angle, "dist_log": dist}
#                 return x, aux
#             else:
#                 return x

#         # 生成anchors 与 strides，拼接各层输出（与原实现一致）
#         self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
#         x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
#         box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

#         a, b = self.dfl(box).chunk(2, 1)
#         a = self.anchors.unsqueeze(0) - a
#         b = self.anchors.unsqueeze(0) + b
#         box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)      
#         det_out = torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)
        
#         if not return_aux:
#             return det_out       
        
#         # 用p5原始特征做回归 
#         p5 = feats[-1]                        # [B, C_p5, H5, W5]
#         g  = self.pool(p5)                    # [B, C_p5, 1, 1]
#         angle = torch.sigmoid(self.angle_head(g)).flatten(1)  # [B,3]
#         dist  = self.dist_head(g).flatten(1)                  # [B,1]
#         aux = {"angle_norm": angle, "dist_log": dist}
#         return det_out, aux
    
#     def initialize_biases(self):
#         for box, cls, s in zip(self.box, self.cls, self.stride):
#             box[-1].bias.data[:] = 1.0
#             cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
        





# class MLPHead(nn.Module):
#     """小型 MLP 头：LayerNorm -> Linear -> GELU -> Dropout -> Linear (+ optional activation)"""
#     def __init__(self, in_dim, hidden_dim, out_dim, p=0.0, out_act: str = None):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(in_dim),
#             nn.Linear(in_dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(p),
#             nn.Linear(hidden_dim, out_dim),
#         )
#         if out_act is None:
#             self.act = nn.Identity()
#         elif out_act == "sigmoid":
#             self.act = nn.Sigmoid()
#         elif out_act == "tanh":
#             self.act = nn.Tanh()
#         else:
#             raise ValueError(f"Unsupported out_act: {out_act}")

#     def forward(self, x):
#         return self.act(self.net(x))


# # 简单检测头
# class SimpleDetHead(nn.Module):
#     """
#     输入:  x_feat (B, L, C), H, W  —— 与 FocalNet.forward_featuremaps 对齐
#     输出:
#       - pred_logits: (B, K, H, W)   类别分数 (sigmoid)
#       - pred_obj   : (B, 1, H, W)   置信度   (sigmoid)
#       - pred_boxes : (B, 4, H, W)   盒子(cx,cy,w,h) 归一化到 0..1 (sigmoid)
#     """
#     def __init__(self, in_channels, num_classes, mid_channels=256):
#         super().__init__() 
#         self.porj = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        
#         # 共享特征塔
#         self.shared = nn.Sequential(
#             nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=True),
#             nn.ReLU(inplace=True),
#         )
        
#         # 分支
#         self.cls_head = nn.Conv2d(mid_channels, num_classes, 1, bias=True) 
#         self.obj_head = nn.Conv2d(mid_channels, 1, 1, bias=True) 
#         self.box_head = nn.Conv2d(mid_channels, 4, 1, bias=True)
        
#         # ---- better init ----
#         nn.init.zeros_(self.cls_head.bias)          # 分类初始不偏置
#         nn.init.constant_(self.obj_head.bias, 0.0)  # obj 不要太低（避免乘没）
#         with torch.no_grad():
#             self.box_head.bias.zero_()
#             # 给 w/h 负偏置，sigmoid(-1)≈0.268，避免 w/h=0 的塌缩
#             self.box_head.bias[2].fill_(-1.0)  # w
#             self.box_head.bias[3].fill_(-1.0)  # h
        
#     def forward(self, x_feat, H, W):
#         # x_feat: (B, L, C) -> (B, C, H, W)
#         B, L, C = x_feat.shape
#         x = x_feat.transpose(1, 2).contiguous().view(B, C, H, W)   
        
#         x = self.porj(x) 
#         x = self.shared(x) 
        
#         pred_logits = torch.sigmoid(self.cls_head(x))  # (B, K, H, W) 
#         pred_obj    = torch.sigmoid(self.obj_head(x))   # (B, 1, H, W)
#         pred_boxes  = torch.sigmoid(self.box_head(x))   # (B, 4, H, W) in [0, 1], (cx,cy,w,h) 
        
#         return pred_logits, pred_obj, pred_boxes



# class YOLO(torch.nn.Module):
#     def __init__(self, width, depth, csp, num_classes):
#         super().__init__()
#         self.net = DarkNet(width, depth, csp)
#         self.fpn = DarkFPN(width, depth, csp)

#         img_dummy = torch.zeros(1, width[0], 256, 256)
#         self.head = Head(num_classes, (width[3], width[4], width[5]))
#         self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
#         self.stride = self.head.stride
#         self.head.initialize_biases()

#     def forward(self, x):
#         x = self.net(x)
#         x = self.fpn(x)
#         return self.head(list(x))

#     def fuse(self):
#         for m in self.modules():
#             if type(m) is Conv and hasattr(m, 'norm'):
#                 m.conv = fuse_conv(m.conv, m.norm)
#                 m.forward = m.fuse_forward
#                 delattr(m, 'norm')
#         return self


# class YOLOMultiTask(torch.nn.Module):
#     def __init__(self, width, depth, csp, num_classes):
#         super().__init__()
#         self.net = DarkNet(width, depth, csp)
#         self.fpn = DarkFPN(width, depth, csp)

#         self.head = MultiTaskHead(num_classes, (width[3], width[4], width[5]))
        
#         # === 用骨干+FPN的特征层尺寸来计算 stride（不要用 head 的输出）===
#         img_dummy = torch.zeros(1, width[0], 256, 256)
#         with torch.no_grad():
#             p3, p4, p5 = self.fpn(self.net(img_dummy))
#         self.head.stride = torch.tensor([256 / p.shape[-2] for p in (p3, p4, p5)])
#         self.stride = self.head.stride
#         self.head.initialize_biases()

#     def forward(self, x, return_aux=True):
#         p3, p4, p5 = self.fpn(self.net(x))
#         out = self.head([p3, p4, p5], return_aux=return_aux)
#         return out


#     def fuse(self):
#         for m in self.modules():
#             if type(m) is Conv and hasattr(m, 'norm'):
#                 m.conv = fuse_conv(m.conv, m.norm)
#                 m.forward = m.fuse_forward
#                 delattr(m, 'norm')
#         return self

# # class YOLOMultiTask(torch.nn.Module):
# #     """
# #     DarkNet + DarkFPN 做骨干
# #     - 图像级：cls / angle_norm(3 in [0,1]) / dist_log(1)
# #     - 可选：检测头（用 p3 的网格特征）
# #     返回字典键名与你现有管线保持一致：
# #       - detect=True:  pred_logits, pred_obj, pred_boxes, feat_size, angle_norm, dist_log[, logits]
# #       - detect=False: logits(可选), angle_norm, dist_log
# #     """
# #     def __init__(self, width, depth, csp,
# #                  num_classes_cls: int = 4,     # 图像级分类类别数
# #                  detect: bool = True,
# #                  det_num_classes: int = None,  # 检测类别数（默认跟图像级相同）
# #                  det_mid_channels: int = 256,
# #                  head_hidden: int = 512,
# #                  head_dropout: float = 0.1,
# #                  keep_img_logits: bool = False # 如果还想要图像级分类logits一起返回
# #                  ):
# #         super().__init__()
# #         # Backbone & FPN 与原 YOLO 一致
# #         self.net = DarkNet(width, depth, csp)
# #         self.fpn = DarkFPN(width, depth, csp)

# #         # 图像级多任务头（对 p5 做 GAP）
# #         E = width[5]                  # p5 的通道数
# #         self.pool = torch.nn.AdaptiveAvgPool2d(1)
# #         self.cls_head   = MLPHead(E, head_hidden, num_classes_cls, p=head_dropout, out_act=None)
# #         self.angle_head = MLPHead(E, head_hidden, 3,             p=head_dropout, out_act="sigmoid")
# #         self.dist_head  = MLPHead(E, head_hidden, 1,             p=head_dropout, out_act=None)
# #         self.keep_img_logits = bool(keep_img_logits)

# #         # 可选检测头：使用 p3（stride=8）的高分辨率特征
# #         self.detect = bool(detect)
# #         if self.detect:
# #             k = det_num_classes if det_num_classes is not None else num_classes_cls
# #             C_p3 = width[3]          # p3 的通道数
# #             self.det_head = SimpleDetHead(in_channels=C_p3, num_classes=k, mid_channels=det_mid_channels)

# #     def forward(self, x):
# #         # 1) Backbone + FPN
# #         p3, p4, p5 = self.fpn(self.net(x))   # p3: [B, C3, H3, W3], p4: ..., p5: ...
# #         B, C3, H3, W3 = p3.shape

# #         # 2) 图像级分支（对 p5 做 GAP）
# #         g = self.pool(p5).flatten(1)         # [B, C5]
# #         angle_norm = self.angle_head(g)      # [B,3] in [0,1]
# #         dist_log   = self.dist_head(g)       # [B,1]
# #         out = {
# #             "angle_norm": angle_norm,
# #             "dist_log": dist_log
# #         }
# #         if self.keep_img_logits:
# #             logits = self.cls_head(g)        # [B,num_classes_cls]
# #             out["logits"] = logits

# #         # 3) 可选检测分支（用 p3）
# #         if self.detect:
# #             tokens = p3.flatten(2).transpose(1, 2)   # [B, L=H3*W3, C3]
# #             pred_logits, pred_obj, pred_boxes = self.det_head(tokens, H3, W3)
# #             out.update({
# #                 "pred_logits": pred_logits,         # (B, K, H3, W3) 
# #                 "pred_obj":    pred_obj,            # (B, 1, H3, W3)  
# #                 "pred_boxes":  pred_boxes,          # (B, 4, H3, W3)  ∈ [0,1] (cx,cy,w,h)
# #                 "feat_size":   (H3, W3)
# #             })
# #         return out

# #     # 如果你需要融合卷积（量化/推理友好），可沿用 YOLO 的 fuse 逻辑
# #     def fuse(self):
# #         for m in self.modules():
# #             if isinstance(m, Conv) and hasattr(m, 'norm'):
# #                 m.conv = fuse_conv(m.conv, m.norm)
# #                 m.forward = m.fuse_forward
# #                 delattr(m, 'norm')
# #         return self



# # ===================== Helper functions for loading =====================

# # ---------- 你已有的工具函数，补充 det_head 前缀 ----------
# def _torch_load_safely(path, map_location="cpu"):
#     try:
#         return torch.load(path, map_location=map_location, weights_only=False)
#     except TypeError:
#         return torch.load(path, map_location=map_location)

# # def _extract_state_dict(obj):
# #     if isinstance(obj, dict):
# #         if "state_dict" in obj and isinstance(obj["state_dict"], (dict,)):
# #             sd = obj["state_dict"]
# #         elif "model" in obj and isinstance(obj["model"], (dict,)):
# #             sd = obj["model"]
# #         else:
# #             if all(isinstance(v, torch.Tensor) for v in obj.values()):
# #                 sd = obj
# #             else:
# #                 raise ValueError("无法从 checkpoint dict 中确定 state_dict，请检查键名（期望 'state_dict' 或 'model'）。")
# #     else:
# #         if hasattr(obj, "state_dict"):
# #             sd = obj.state_dict()
# #         else:
# #             raise ValueError(f"未知的 checkpoint 类型：{type(obj)}，无法提取 state_dict。")
# #     return sd
# def _extract_state_dict(obj):
#     """
#     从多种常见 checkpoint 结构中提取 state_dict。
#     兼容：纯 state_dict、{'state_dict': ...}、{'model': ...}、{'ema': module/...}、以及常见别名键。
#     """
#     import torch
#     from collections import OrderedDict

#     # 如果是“整个模型/EMA 模型对象”
#     if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")) and not isinstance(obj, (dict,)):
#         return obj.state_dict()

#     if isinstance(obj, (dict, OrderedDict)):
#         # 1) 首选常见键
#         candidate_keys = [
#             "state_dict", "model", "ema", "model_ema",
#             "ema_state_dict", "model_state",
#             "net", "network", "weights", "params",  # 兼容各种训练脚本
#         ]
#         for k in candidate_keys:
#             if k in obj:
#                 v = obj[k]
#                 # a) 直接是 dict 且值基本都是 Tensor
#                 if isinstance(v, (dict, OrderedDict)) and all(isinstance(x, torch.Tensor) for x in v.values()):
#                     return v
#                 # b) 是模块对象
#                 if hasattr(v, "state_dict") and callable(getattr(v, "state_dict")):
#                     return v.state_dict()

#         # 2) 直接就是纯 state_dict？
#         if all(isinstance(v, torch.Tensor) for v in obj.values()):
#             return obj

#         # 3) 再做一层浅层扫描：某个 value 本身是“看起来像 state_dict”的 dict
#         for k, v in obj.items():
#             if isinstance(v, (dict, OrderedDict)) and v:
#                 if all(isinstance(x, torch.Tensor) for x in v.values()):
#                     return v
#                 # 某些脚本把 state_dict 再包了一层 model/ema 对象字段
#                 if hasattr(v, "state_dict") and callable(getattr(v, "state_dict")):
#                     return v.state_dict()

#         # 4) 实在不行，给个提示
#         raise ValueError(
#             "无法从 checkpoint dict 中确定 state_dict，可用键包括："
#             + ", ".join(map(str, obj.keys()))
#             + "。请检查权重文件的结构。"
#         )

#     # 其它类型：尝试当作模块
#     if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
#         return obj.state_dict()

#     raise ValueError(f"未知的 checkpoint 类型：{type(obj)}，无法提取 state_dict。")

# def _strip_prefix(sd, prefixes=("module.", "model.")):
#     new_sd = {}
#     for k, v in sd.items():
#         nk = k
#         for p in prefixes:
#             if nk.startswith(p):
#                 nk = nk[len(p):]
#         new_sd[nk] = v
#     return new_sd

# def _filter_heads_for_pretrain(sd):
#     """
#     预训练阶段通常不加载任务头（图像级 + 检测头）。
#     """
#     drop_prefixes = ("cls_head.", "angle_head.", "dist_head.", "det_head.")
#     return {k: v for k, v in sd.items() if not any(k.startswith(p) for p in drop_prefixes)}

# def _shape_compatible_only(model, sd):
#     msd = model.state_dict()
#     filtered, skipped = {}, []
#     for k, v in sd.items():
#         if k in msd and msd[k].shape == v.shape:
#             filtered[k] = v
#         else:
#             skipped.append(k)
#     return filtered, skipped

# def _load_checkpoint_to_model(model, ckpt_path, strict=False, drop_heads=False, tag=""):
#     obj = _torch_load_safely(ckpt_path, map_location="cpu")
#     sd = _extract_state_dict(obj)
#     sd = _strip_prefix(sd)
#     if drop_heads:
#         sd = _filter_heads_for_pretrain(sd)
#     sd, skipped = _shape_compatible_only(model, sd)
#     missing, unexpected = model.load_state_dict(sd, strict=strict)

#     print(f"[{tag}] Loaded from: {ckpt_path}")
#     print(f"[{tag}] Loaded keys: {len(sd)} | Skipped (shape mismatch): {len(skipped)}")
#     if missing:
#         print(f"[{tag}] Missing keys ({len(missing)}): {sorted(missing)[:8]}{' ...' if len(missing)>8 else ''}")
#     if unexpected:
#         print(f"[{tag}] Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:8]}{' ...' if len(unexpected)>8 else ''}")

# # ---------- YOLOMultiTask 变体解析 ----------
# def _resolve_yolo_cfg(args):
#     """
#     支持三种方式指定骨干宽度/深度：
#       1) 直接给 args.width / args.depth / args.csp
#       2) 指定 args.size 或 args.variant in {n,t,s,m,l,x}
#       3) 默认 's'
#     返回 (width, depth, csp)
#     """
#     if hasattr(args, "width") and hasattr(args, "depth") and hasattr(args, "csp") and args.width and args.depth and args.csp:
#         return args.width, args.depth, args.csp

#     size = (getattr(args, "size", None) or getattr(args, "variant", "s")).lower()
#     if size == "n":
#         csp   = [False, True]
#         depth = [1, 1, 1, 1, 1, 1]
#         width = [3, 16, 32, 64, 128, 256]
#     elif size == "t":
#         csp   = [False, True]
#         depth = [1, 1, 1, 1, 1, 1]
#         width = [3, 24, 48, 96, 192, 384]
#     elif size == "s":
#         csp   = [False, True]
#         depth = [1, 1, 1, 1, 1, 1]
#         width = [3, 32, 64, 128, 256, 512]
#     elif size == "m":
#         csp   = [True, True]
#         depth = [1, 1, 1, 1, 1, 1]
#         width = [3, 64, 128, 256, 512, 512]
#     elif size == "l":
#         csp   = [True, True]
#         depth = [2, 2, 2, 2, 2, 2]
#         width = [3, 64, 128, 256, 512, 512]
#     elif size == "x":
#         csp   = [True, True]
#         depth = [2, 2, 2, 2, 2, 2]
#         width = [3, 96, 192, 384, 768, 768]
#     else:
#         raise ValueError(f"Unknown YOLO size/variant: {size}")
#     return width, depth, csp

# # ===================== Modified build_model =====================
# def build_model(args):
#     # 解析设备
#     device = torch.device(getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu"))

#     # 解析 YOLO 变体
#     width, depth, csp = _resolve_yolo_cfg(args)

#     # 读取多任务/检测头超参
#     num_classes       = int(getattr(args, "num_classes", 4))          # 图像级类别数（也可用于 det_num_classes）
#     detect            = bool(getattr(args, "detect", True))
#     det_num_classes   = getattr(args, "det_num_classes", None) or num_classes
#     det_mid_channels  = int(getattr(args, "det_mid_channels", 256))
#     head_hidden       = int(getattr(args, "head_hidden", 512))
#     head_dropout      = float(getattr(args, "head_dropout", 0.1))
#     keep_img_logits   = bool(getattr(args, "keep_img_logits", False))

#     # 构建 YOLOMultiTask
#     model = YOLOMultiTask(
#         width=width,
#         depth=depth,
#         csp=csp,
#         num_classes_cls=num_classes,
#         # detect=detect,
#         # det_num_classes=det_num_classes,
#         # det_mid_channels=det_mid_channels,
#         # head_hidden=head_hidden,
#         # head_dropout=head_dropout,
#         # keep_img_logits=keep_img_logits,
#     ).to(device)

#     # 模式与权重加载
#     mode = str(getattr(args, "mode", "train")).lower()
#     pretrain = bool(getattr(args, "pretrain", False))

#     if mode in ("train", "training"):
#         if pretrain:
#             pretrain_path = getattr(args, "pretrain_path", None)
#             if pretrain_path is None or not os.path.isfile(pretrain_path):
#                 print("[Pretrain] 未提供有效的 args.pretrain_path，跳过加载预训练权重。")
#             else:
#                 # 训练：通常跳过所有 head（包含 det_head.*）
#                 _load_checkpoint_to_model(
#                     model, pretrain_path,
#                     strict=False, drop_heads=True, tag="Pretrain"
#                 )
#         model.train()

#     elif mode in ("test", "eval", "evaluation", "inference"):
#         ckpt = getattr(args, "ckpt", None) or getattr(args, "checkpoint", None) or getattr(args, "weights", None)
#         if ckpt is None or not os.path.isfile(ckpt):
#             print("[Test] 未提供有效的测试权重路径（args.ckpt / args.checkpoint / args.weights），将以随机初始化权重进行评估。")
#         else:
#             # 测试：加载完整模型（包含头）；strict=False 以兼容微小不一致
#             _load_checkpoint_to_model(
#                 model, ckpt,
#                 strict=False, drop_heads=False, tag="Test/Eval"
#             )
#         model.eval()

#     else:
#         print(f"[Warn] 未识别的 args.mode='{getattr(args, 'mode', None)}'，默认按训练模式处理。")
#         model.train()

#     return model


        

# def yolo_v11_n(num_classes: int = 80):
#     csp = [False, True]
#     depth = [1, 1, 1, 1, 1, 1]
#     width = [3, 16, 32, 64, 128, 256]
#     return YOLO(width, depth, csp, num_classes)


# def yolo_v11_t(num_classes: int = 80):
#     csp = [False, True]
#     depth = [1, 1, 1, 1, 1, 1]
#     width = [3, 24, 48, 96, 192, 384]
#     return YOLO(width, depth, csp, num_classes)


# def yolo_v11_s(num_classes: int = 80):
#     csp = [False, True]
#     depth = [1, 1, 1, 1, 1, 1]
#     width = [3, 32, 64, 128, 256, 512]
#     return YOLO(width, depth, csp, num_classes)


# def yolo_v11_m(num_classes: int = 80):
#     csp = [True, True]
#     depth = [1, 1, 1, 1, 1, 1]
#     width = [3, 64, 128, 256, 512, 512]
#     return YOLO(width, depth, csp, num_classes)


# def yolo_v11_l(num_classes: int = 80):
#     csp = [True, True]
#     depth = [2, 2, 2, 2, 2, 2]
#     width = [3, 64, 128, 256, 512, 512]
#     return YOLO(width, depth, csp, num_classes)


# def yolo_v11_x(num_classes: int = 80):
#     csp = [True, True]
#     depth = [2, 2, 2, 2, 2, 2]
#     width = [3, 96, 192, 384, 768, 768]
#     return YOLO(width, depth, csp, num_classes)


import math

import torch

from utils.util import make_anchors


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


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box,torch.nn.SiLU(), k=3, p=1),
                                                           Conv(box, box,torch.nn.SiLU(), k=3, p=1),
                                                           torch.nn.Conv2d(box, out_channels=4 * self.ch,
                                                                           kernel_size=1)) for x in filters)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, x, torch.nn.SiLU(), k=3, p=1, g=x),
                                                           Conv(x, cls, torch.nn.SiLU()),
                                                           Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
                                                           Conv(cls, cls, torch.nn.SiLU()),
                                                           torch.nn.Conv2d(cls, out_channels=self.nc,
                                                                           kernel_size=1)) for x in filters)

    def forward(self, x):
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        if self.training:
            return x

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        for box, cls, s in zip(self.box, self.cls, self.stride):
            # box
            box[-1].bias.data[:] = 1.0
            # cls (.01 objects, 80 classes, 640 image)
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes):
        super().__init__()
        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v11_n(num_classes: int = 80):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_t(num_classes: int = 80):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 24, 48, 96, 192, 384]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_s(num_classes: int = 80):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_m(num_classes: int = 80):
    csp = [True, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_l(num_classes: int = 80):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_x(num_classes: int = 80):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    return YOLO(width, depth, csp, num_classes)