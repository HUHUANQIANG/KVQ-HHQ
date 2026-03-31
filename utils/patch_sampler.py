"""
utils/patch_sampler.py

CPU 端工具：基于低分辨率 Saliency Map 在原始 4K 视频中
进行坐标映射、Patch 裁剪，实现 Saliency-Guided Patch Sampling。

核心函数：
    get_saliency_guided_patches(
        saliency_map_lr,   # Tensor[B, 1, H_s, W_s]，低分辨率 Saliency Map（已 detach 到 CPU）
        video_path,        # str 或 List[str]，原始 4K 视频路径（单样本传 str，批量传 List[str]）
        frame_indices,     # List[int] 或 List[List[int]]，需要裁剪的帧索引（不能为空）
        orig_h,            # int，原始视频高度（默认 2160）
        orig_w,            # int，原始视频宽度（默认 3840）
        patch_size,        # int，裁剪 patch 的边长，默认 256
        top_k,             # int，Top-K Saliency 采样数，默认 6
        rand_k,            # int，随机采样数，默认 2
    ) -> Tuple[Tensor, Tensor]
        返回：
            hr_patches      : Tensor[B, T, num_patches, C, patch_size, patch_size]
            saliency_weights: Tensor[B, num_patches]  对应 patch 的 Saliency 权重（已归一化）
"""

import random
import numpy as np
import torch
import decord
from decord import VideoReader, cpu

decord.bridge.set_bridge("torch")

# ImageNet 均值 / 标准差，用于 HR patch 归一化
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _normalize_patch(patch: torch.Tensor) -> torch.Tensor:
    """将 [0,1] 范围的 patch 做 ImageNet 归一化。

    Args:
        patch: Tensor[C, H, W]，值域 [0, 1]。

    Returns:
        Tensor[C, H, W]，已归一化。
    """
    return (patch - _IMAGENET_MEAN) / _IMAGENET_STD


def _get_patch_box(cy: int, cx: int, patch_size: int, img_h: int, img_w: int):
    """根据中心点坐标计算 patch 的边界框，并处理越界情况。

    Args:
        cy (int): 中心点纵坐标（行）。
        cx (int): 中心点横坐标（列）。
        patch_size (int): patch 边长。
        img_h (int): 图像高度。
        img_w (int): 图像宽度。

    Returns:
        Tuple[int, int, int, int]: (y1, y2, x1, x2) 边界框坐标。
    """
    half = patch_size // 2
    y1 = cy - half
    y2 = cy + half
    x1 = cx - half
    x2 = cx + half

    # 左/上越界：整体右/下移
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x1 < 0:
        x2 -= x1
        x1 = 0

    # 右/下越界：整体左/上移
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w

    # 再次 clamp 防止因图像过小导致的负值
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(img_h, y2)
    x2 = min(img_w, x2)

    return y1, y2, x1, x2


def _saliency_to_patch_centers(
    sal_map: torch.Tensor,
    orig_h: int,
    orig_w: int,
    top_k: int,
    rand_k: int,
) -> tuple:
    """将低分辨率 Saliency Map 的激活点映射到原始分辨率坐标。

    策略：
        - Top-K 高 Saliency 位置
        - rand_k 随机位置（防止采样偏置）

    Args:
        sal_map (Tensor[H_s, W_s]): 单帧单通道 Saliency Map。
        orig_h (int): 原始视频高度。
        orig_w (int): 原始视频宽度。
        top_k (int): Top-K Saliency 点数量。
        rand_k (int): 随机采样点数量。

    Returns:
        centers (List[Tuple[int,int]]): 中心点 (cy, cx) 列表，长度为 top_k + rand_k。
        weights (List[float]): 每个中心点对应的 Saliency 权重。
    """
    H_s, W_s = sal_map.shape
    # 展平后取 Top-K
    flat = sal_map.view(-1)
    num_positions = flat.numel()
    actual_k = min(top_k, num_positions)
    topk_vals, topk_flat_idx = torch.topk(flat, k=actual_k)

    # 换算回 (row, col) 索引
    topk_rows = (topk_flat_idx // W_s).numpy()
    topk_cols = (topk_flat_idx % W_s).numpy()
    topk_vals = topk_vals.numpy().tolist()

    # 映射到原始分辨率中心坐标
    scale_h = orig_h / H_s
    scale_w = orig_w / W_s

    centers = []
    weights = []
    for r, c, v in zip(topk_rows, topk_cols, topk_vals):
        cy = int((r + 0.5) * scale_h)
        cx = int((c + 0.5) * scale_w)
        cy = min(cy, orig_h - 1)
        cx = min(cx, orig_w - 1)
        centers.append((cy, cx))
        weights.append(float(v))

    # 随机采样补充
    for _ in range(rand_k):
        ry = random.randint(0, orig_h - 1)
        rx = random.randint(0, orig_w - 1)
        centers.append((ry, rx))
        # 随机 patch 权重取 Saliency Map 均值（保守估计）
        weights.append(float(sal_map.mean().item()))

    return centers, weights


def get_saliency_guided_patches(
    saliency_map_lr: torch.Tensor,
    video_path: str,
    frame_indices: list,
    orig_h: int = 2160,
    orig_w: int = 3840,
    patch_size: int = 256,
    top_k: int = 6,
    rand_k: int = 2,
) -> tuple:
    """基于低分辨率 Saliency Map，从原始 4K 视频中裁剪高分辨率 Patches。

    整个操作在 CPU 上完成，不占用 GPU 显存。

    Args:
        saliency_map_lr (Tensor[B, 1, H_s, W_s]): 低分辨率 Saliency Map，
            已 detach 到 CPU。每个 batch 元素对应一段视频片段（T 帧共享同一 map）。
        video_path (str 或 List[str]): 原始视频路径。
            当 B=1 时可传 str；B>1 时传 List[str]，长度需等于 B。
        frame_indices (List[int] 或 List[List[int]]): 帧索引列表。
            当 B=1 时可传 List[int]；B>1 时传 List[List[int]]。
        orig_h (int): 原始视频高度，默认 2160（4K）。
        orig_w (int): 原始视频宽度，默认 3840（4K）。
        patch_size (int): Patch 边长（正方形），默认 256。
        top_k (int): 从 Saliency Map 中选取的 Top-K 高激活位置数，默认 6。
        rand_k (int): 额外随机采样的 Patch 数（防止偏置），默认 2。

    Returns:
        hr_patches (Tensor[B, T, num_patches, C, patch_size, patch_size]):
            裁剪后的高分辨率 Patch，归一化至 ImageNet 标准。
        saliency_weights (Tensor[B, num_patches]):
            每个 Patch 对应的 Saliency 权重（已做 softmax 归一化）。
    """
    # 统一输入格式（支持单样本和 batch）
    B = saliency_map_lr.shape[0]
    if isinstance(video_path, str):
        video_paths = [video_path] * B
    else:
        video_paths = list(video_path)

    if not frame_indices:
        raise ValueError("frame_indices 不能为空列表")
    # 判断是 List[int]（单样本）还是 List[List[int]]（batch）
    first_elem = frame_indices[0]
    if isinstance(first_elem, (int, np.integer)):
        frame_indices_list = [list(frame_indices)] * B
    else:
        frame_indices_list = [list(fi) for fi in frame_indices]

    num_patches = top_k + rand_k
    T = len(frame_indices_list[0])

    all_patches = []        # List[B] -> Tensor[T, num_patches, C, patch_size, patch_size]
    all_weights = []        # List[B] -> Tensor[num_patches]

    for b in range(B):
        sal_map_b = saliency_map_lr[b, 0]  # [H_s, W_s]
        vpath = video_paths[b]
        findices = frame_indices_list[b]

        # ------------------------------------------------------------------
        # 1. Saliency Map -> 原始分辨率坐标（所有帧共享同一 Saliency Map）
        # ------------------------------------------------------------------
        centers, weights = _saliency_to_patch_centers(
            sal_map_b, orig_h, orig_w, top_k, rand_k
        )
        # Softmax 归一化权重
        w_tensor = torch.tensor(weights, dtype=torch.float32)
        w_tensor = torch.softmax(w_tensor, dim=0)  # [num_patches]
        all_weights.append(w_tensor)

        # ------------------------------------------------------------------
        # 2. 从硬盘读取对应帧并裁剪 Patch
        # ------------------------------------------------------------------
        vr = VideoReader(vpath, ctx=cpu(0))
        frames_raw = vr.get_batch(findices)  # [T, H, W, C]，uint8
        del vr

        frame_patches = []  # List[T] -> Tensor[num_patches, C, patch_size, patch_size]
        for t in range(T):
            frame = frames_raw[t]  # [H, W, C]，uint8 Tensor

            # 获取实际视频尺寸（防止与 orig_h/orig_w 假设不符）
            actual_h, actual_w = frame.shape[0], frame.shape[1]

            patch_tensors = []
            for cy, cx in centers:
                # 根据实际视频尺寸重新 clamp 坐标
                cy_clamped = min(cy, actual_h - 1)
                cx_clamped = min(cx, actual_w - 1)
                y1, y2, x1, x2 = _get_patch_box(cy_clamped, cx_clamped, patch_size, actual_h, actual_w)

                patch = frame[y1:y2, x1:x2, :]  # [patch_size, patch_size, C]
                # 转为 float [0,1]，并变换轴为 [C, H, W]
                patch = patch.float() / 255.0
                patch = patch.permute(2, 0, 1)  # [C, H, W]

                # 若裁剪结果尺寸不足（视频本身分辨率低于 patch_size），做双线性上采样
                if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                    patch = torch.nn.functional.interpolate(
                        patch.unsqueeze(0),
                        size=(patch_size, patch_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

                patch = _normalize_patch(patch)  # [C, patch_size, patch_size]
                patch_tensors.append(patch)

            # [num_patches, C, patch_size, patch_size]
            frame_patches.append(torch.stack(patch_tensors, dim=0))

        # [T, num_patches, C, patch_size, patch_size]
        all_patches.append(torch.stack(frame_patches, dim=0))

    # 堆叠成 batch
    hr_patches = torch.stack(all_patches, dim=0)          # [B, T, num_patches, C, ps, ps]
    saliency_weights = torch.stack(all_weights, dim=0)    # [B, num_patches]

    return hr_patches, saliency_weights
