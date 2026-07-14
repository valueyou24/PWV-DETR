"""
backbone 特征图提取脚本
- 加载 PRV 模型训练权重
- 使用 hook 捕获 backbone 各层输出
- 可视化特征图（通道平均响应热力图 + 前N个通道的网格图）
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# 添加项目路径
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics.nn.tasks import RTDETRDetectionModel


def preprocess_image(img_path, target_size=640):
    """
    对输入图像做与 RTDETR predictor 一致的预处理，返回 tensor (1, 3, H, W)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # LetterBox + resize（与 RTDETR 推理时的预处理对齐：正方形、scaleFill）
    h0, w0 = img.shape[:2]
    scale = target_size / max(h0, w0)
    if scale != 1:
        new_h, new_w = int(round(h0 * scale)), int(round(w0 * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # pad 到正方形
    pad_h, pad_w = target_size - new_h, target_size - new_w
    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # HWC -> CHW, uint8 -> float32, normalize to [0,1]
    tensor = img.transpose(2, 0, 1)[None]  # (1, 3, H, W)
    tensor = torch.from_numpy(tensor).float() / 255.0
    return tensor, (h0, w0), img  # (原始高, 原始宽), 处理后的 numpy 图


def register_backbone_hooks(model, backbone_indices):
    """
    注册 forward hook 到指定的 backbone 层
    backbone_indices: list, 如 [5, 6, 7] 对应 P3, P4, P5
    返回: dict {index: feature_tensor}
    """
    features = {}

    def hook_fn_factory(idx):
        def hook_fn(module, input, output):
            features[idx] = output.detach().cpu()
        return hook_fn

    handles = []
    for idx in backbone_indices:
        module = model.model[idx]
        handle = module.register_forward_hook(hook_fn_factory(idx))
        handles.append(handle)

    return features, handles


def visualize_channel_mean_heatmap(feature_map, save_path, title="Features"):
    """
    对所有通道取均值，得到 (H, W) 的空间响应热力图
    """
    if feature_map.dim() == 4:
        fmap = feature_map[0].numpy()  # (C, H, W)
    else:
        fmap = feature_map.numpy()

    mean_map = fmap.mean(axis=0)  # (H, W)，所有通道的均值响应

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(mean_map, cmap='jet', interpolation='bilinear')
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [热力图] 已保存: {save_path}")


def visualize_top_channels_grid(feature_map, save_path, title="Feature Channels", n_cols=8, top_n=32):
    """
    取响应最强的 top_n 个通道，按网格排列显示
    """
    if feature_map.dim() == 4:
        fmap = feature_map[0].numpy()  # (C, H, W)
    else:
        fmap = feature_map.numpy()

    C = fmap.shape[0]
    top_n = min(top_n, C)

    # 按通道的全局均值排序，取响应最强的 top_n 个通道
    channel_means = fmap.reshape(C, -1).mean(axis=1)
    top_indices = np.argsort(channel_means)[::-1][:top_n]

    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = axes.flatten() if top_n > 1 else [axes]

    for i in range(top_n):
        ch_idx = top_indices[i]
        axes[i].imshow(fmap[ch_idx], cmap='jet')
        axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(top_n, len(axes)):
        axes[i].axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"  [通道图] 已保存: {save_path}")


def overlay_heatmap_on_image(feature_map, original_img, save_path, title="Overlay", alpha=0.5):
    """
    将特征热力图叠加到原图上
    original_img: (H, W, 3) numpy array, RGB, uint8
    """
    if feature_map.dim() == 4:
        fmap = feature_map[0].numpy()
    else:
        fmap = feature_map.numpy()

    mean_map = fmap.mean(axis=0)  # (H, W)

    # 将均值图 resize 到原图大小
    h_orig, w_orig = original_img.shape[:2]
    heatmap = cv2.resize(mean_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    # 归一化到 [0, 255]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 叠加
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)

    cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"  [叠加图] 已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="提取 backbone 特征图")
    parser.add_argument('--weights', type=str, required=True,
                        default='runs/train/exp-PRV-WiseIoU/weights/best.pt',
                        help='训练后的权重文件路径 (e.g. runs/train/exp/weights/best.pt)')
    parser.add_argument('--image', type=str, required=True,
                        default='dataset/mydata/images/test/001000.jpg',
                        help='输入图像路径')
    parser.add_argument('--model-cfg', type=str,
                        default='ultralytics/cfg/models/PConv_rep.yaml',
                        help='模型 YAML 配置文件')
    parser.add_argument('--output-dir', type=str, default='./backbone_features_pwv',
                        help='输出目录')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='推理图像尺寸（正方形）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备: cuda 或 cpu')
    parser.add_argument('--top-channels', type=int, default=32,
                        help='通道网格图中显示的通道数')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 1. 加载模型和权重 ==========
    print(f"[INFO] 加载模型配置: {args.model_cfg}")
    model = RTDETRDetectionModel(cfg=args.model_cfg, ch=3, nc=1, verbose=False)
    model.eval()

    print(f"[INFO] 加载权重: {args.weights}")
    weights = torch.load(args.weights, map_location='cpu', weights_only=False)
    # 处理不同格式的权重文件
    if 'model' in weights:
        state_dict = weights['model'].float().state_dict() if hasattr(weights['model'], 'state_dict') else weights['model']
    elif 'ema' in weights:
        state_dict = weights['ema'].float().state_dict() if hasattr(weights['ema'], 'state_dict') else weights['ema']
    else:
        state_dict = weights
    model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] 模型加载完毕")

    model.to(device)

    # ========== 2. 预处理图像 ==========
    print(f"[INFO] 预处理图像: {args.image}")
    tensor, (h0, w0), processed_np_img = preprocess_image(args.image, target_size=args.imgsz)
    tensor = tensor.to(device)

    # 保存预处理后的图像，供叠加参考
    preproc_save = output_dir / 'input_preprocessed.jpg'
    cv2.imwrite(str(preproc_save), cv2.cvtColor(processed_np_img, cv2.COLOR_RGB2BGR))
    print(f"  [预处理图] 已保存: {preproc_save}")

    # ========== 3. 注册 backbone hook ==========
    # PRV.yaml backbone P3/P4/P5 (对应 YAML 索引 5/6/7, 1-based 编号 6/7/8):
    backbone_indices = [5, 6, 7]
    names = {5: 'P3_stride8_128ch', 6: 'P4_stride16_256ch', 7: 'P5_stride32_512ch'}
    features, handles = register_backbone_hooks(model, backbone_indices)

    # ========== 4. 前向推理 ==========
    print(f"[INFO] 执行前向推理...")
    with torch.no_grad():
        _ = model.predict(tensor)

    # ========== 5. 可视化各层 Top-1 通道图 ==========
    print(f"\n[INFO] 开始输出 backbone 各层 Top-1 通道图...")
    print(f"=" * 60)

    for idx in backbone_indices:
        if idx not in features:
            print(f"[WARN] 未捕获到第{idx}层特征!")
            continue

        fmap = features[idx]  # (1, C, H, W)
        C = fmap.shape[1]
        print(f"\n--- 第{idx}层 ({names[idx]}): C={C}, shape={list(fmap.shape)} ---")

        # 输出响应最强的通道的网格图（top 1）
        grid_path = output_dir / f'layer{idx}_{names[idx]}_top1.png'
        visualize_top_channels_grid(fmap, grid_path,
                                    title=f"Layer {idx} - {names[idx]} - Top 1 Channel",
                                    n_cols=1, top_n=1)

    # ========== 6. 清理 ==========
    for h in handles:
        h.remove()

    print(f"\n{'=' * 60}")
    print(f"[DONE] 所有特征图已保存至: {output_dir.absolute()}")
    print(f"输出文件:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
