from types import SimpleNamespace as NS


# ===== 任务相关 =====
num_classes = 4                     # B / PP / SP / T
modelname   = 'yolov11_multitask'      # 如果你用注册表；若直接 import，可忽略
data_root = "/mnt/data/pxy/Dataset/racketpose/data"
mode = "train"                    # train / val / test
img_size = 224                    # 输入图像尺寸（正方形）


# ===== 优化器 & 训练 =====
lr = 1e-5
weight_decay = 1e-5
batch_size = 64                  # 视显存调整
epochs = 200
lr_drop = 40                        # 或使用多步/OneCycle都可
save_checkpoint_interval = 1
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = False
lr_drop_list = [40, 48]
device = "cuda"
mm_to_m = False                     # 距离单位：毫米转米
seed = 42
save_dir = "runs/yolov11_multitask"
exp_name = "yolov11"
log_dir = None
fp16 = False
num_workers = 4
acc_thresholds = (1.0, 2.0, 5.0, 10.0)
normalize_boxes = True 



# EMA（可选）
use_ema = False
ema_decay = 0.9997
ema_epoch = 0



model = NS(
    size="n",
    num_classes = 4,        # 图像级类别数（也可用于 det_num_classes）
    detect = True,
    det_num_classes = 4,
    det_mid_channels = 256,
    head_hidden = 512,
    head_dropout = 0.1,
    keep_img_logits = False,
    pretrain = False,
    pretrain_path = "/mnt/data/pxy/YOLOv11-pt-master/weights/pretrain/v11_n.pt",
    mode="train",
    ckpt="",
)
