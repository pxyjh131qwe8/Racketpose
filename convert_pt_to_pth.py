# import torch
# import os
# import argparse

# from nets.nn import YOLO

# def convert_one(input_path, output_path):
#     # 在可信环境下，用不安全方式加载一次
#     obj = torch.load(input_path, map_location="cpu", weights_only=False)

#     # 提取 state_dict
#     if hasattr(obj, "state_dict"):
#         state_dict = obj.state_dict()
#     elif isinstance(obj, dict):
#         if "state_dict" in obj:
#             state_dict = obj["state_dict"]
#         elif "model" in obj:
#             state_dict = obj["model"]
#         else:
#             state_dict = obj
#     else:
#         raise ValueError(f"无法从 {input_path} 提取 state_dict，类型是 {type(obj)}")

#     # 保存为纯权重
#     torch.save(state_dict, output_path)
#     print(f"✅ 已转换: {input_path} → {output_path}")

# def batch_convert(folder):
#     for fname in os.listdir(folder):
#         if fname.endswith(".pt"):
#             input_path = os.path.join(folder, fname)
#             base, _ = os.path.splitext(fname)
#             output_path = os.path.join(folder, base + ".pth")
#             try:
#                 convert_one(input_path, output_path)
#             except Exception as e:
#                 print(f"❌ 跳过 {input_path}, 错误: {e}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--folder", default="/data1/wangqiurui/pxy/FSTD/weights/backbone2D/YOLOv11", help="包含 .pt 文件的目录")
#     args = parser.parse_args()

#     batch_convert(args.folder)

import torch
import os
import argparse

def convert_one(input_path, output_path):
    # 不安全加载一次（因为原始 .pt 里有完整模型，必须这样）
    obj = torch.load(input_path, map_location="cpu", weights_only=False)

    # 统一提取出 state_dict
    if hasattr(obj, "state_dict"):
        state_dict = obj.state_dict()
    elif isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state_dict = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            state_dict = obj["model"]
        else:
            # 如果就是一个字典且值都是张量，说明已经是 state_dict
            state_dict = {k: v for k, v in obj.items() if torch.is_tensor(v)}
    else:
        raise ValueError(f"无法从 {input_path} 提取 state_dict，类型是 {type(obj)}")

    # ✅ 只保存权重，不保存模型对象
    torch.save(state_dict, output_path)
    print(f"✅ 已转换为纯权重: {input_path} → {output_path}")

def batch_convert(folder):
    for fname in os.listdir(folder):
        if fname.endswith(".pt"):
            input_path = os.path.join(folder, fname)
            base, _ = os.path.splitext(fname)
            output_path = os.path.join(folder, base + ".pth")
            try:
                convert_one(input_path, output_path)
            except Exception as e:
                print(f"❌ 跳过 {input_path}, 错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        default="/mnt/data/pxy/YOLOv11-pt-master/weights/pretrain",
        help="包含 .pt 文件的目录"
    )
    args = parser.parse_args()

    batch_convert(args.folder)
