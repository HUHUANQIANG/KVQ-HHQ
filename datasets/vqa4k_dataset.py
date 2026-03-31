"""
datasets/vqa4k_dataset.py

轻量级 4K VQA 数据集加载器。
DataLoader 只提供低分辨率帧（224×224）和原始视频访问句柄，
不将完整 4K Tensor 加载到内存，以满足显存管理要求。

__getitem__ 返回字典：
    {
        "lr_frames":     Tensor[C, T, 224, 224],  # 下采样后的低分辨率帧，用于 saliency 网络
        "video_path":    str,                      # 原始视频绝对路径（供 patch sampler 使用）
        "frame_indices": List[int],                # 已采样的帧索引列表
        "mos":           float                     # 主观质量分数（Mean Opinion Score）
    }
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import decord
from decord import VideoReader, cpu

decord.bridge.set_bridge("torch")

# ImageNet 均值 / 标准差，用于低分辨率帧的归一化
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class VQA4KDataset(Dataset):
    """面向 4K 视频质量评估的数据集。

    Args:
        anno_file (str): 标注文件路径，每行格式为 "<video_filename>,<mos>"。
        data_prefix (str): 视频文件所在的根目录。
        clip_len (int): 每个视频片段采样的帧数，默认 8。
        frame_interval (int): 帧采样间隔（步长），默认 2。
        lr_size (int): 低分辨率帧的边长（正方形），默认 224。
        phase (str): 数据集阶段，"train" 或 "test"。
        seed (int): 随机种子，用于训练时的随机帧采样。
    """

    def __init__(
        self,
        anno_file: str,
        data_prefix: str,
        clip_len: int = 8,
        frame_interval: int = 2,
        lr_size: int = 224,
        phase: str = "train",
        seed: int = 42,
    ):
        super().__init__()
        self.anno_file = anno_file
        self.data_prefix = data_prefix
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.lr_size = lr_size
        self.phase = phase
        self.seed = seed

        # 使用固定种子的随机数生成器，保证训练采样可复现
        self._rng = np.random.RandomState(seed)

        # 低分辨率帧的预处理变换
        self.lr_transform = T.Compose([
            T.Resize((lr_size, lr_size)),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        self.video_infos = self._load_annotations()

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _load_annotations(self):
        """解析标注文件，返回 video_info 列表。

        支持两种常见格式：
            1. "<video_filename>,<extra1>,<extra2>,<mos>"  （4 列，如 LSVQ）
            2. "<video_filename>,<mos>"                    （2 列，简化格式）
        """
        video_infos = []
        with open(self.anno_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 4:
                    filename, mos = parts[0], float(parts[3])
                elif len(parts) == 2:
                    filename, mos = parts[0], float(parts[1])
                else:
                    # 格式不符合预期，跳过
                    continue
                video_infos.append(
                    dict(
                        filename=os.path.join(self.data_prefix, filename),
                        mos=mos,
                    )
                )
        return video_infos

    def _sample_frame_indices(self, total_frames: int) -> list:
        """从视频中均匀采样帧索引。

        训练阶段在各均匀区间内随机取帧，测试阶段取区间中心帧，
        保证帧分布的均匀性并避免边界越界。

        Args:
            total_frames (int): 视频总帧数。

        Returns:
            List[int]: 长度为 clip_len 的帧索引列表。
        """
        # 所需的时间跨度
        span = (self.clip_len - 1) * self.frame_interval + 1

        if total_frames <= span:
            # 视频较短：按步长均匀采，不足则最后一帧补齐
            indices = list(range(0, total_frames, self.frame_interval))
            while len(indices) < self.clip_len:
                indices.append(indices[-1])
            indices = indices[: self.clip_len]
        else:
            # 视频较长：将可用帧段均匀分成 clip_len 份，各自内部随机/中心取样
            seg_size = (total_frames - span) // self.clip_len
            if self.phase == "train":
                offsets = self._rng.randint(0, max(1, seg_size), size=self.clip_len)
                starts = [i * seg_size + int(offsets[i]) for i in range(self.clip_len)]
            else:
                starts = [i * seg_size for i in range(self.clip_len)]
            indices = [s + i * self.frame_interval for i, s in enumerate(starts)]
            # 保证不越界
            indices = [min(idx, total_frames - 1) for idx in indices]

        return indices

    def _load_lr_frames(self, video_path: str, frame_indices: list) -> torch.Tensor:
        """使用 decord 按帧索引读取帧，并下采样到低分辨率。

        Args:
            video_path (str): 视频文件路径。
            frame_indices (list): 帧索引列表。

        Returns:
            Tensor[C, T, lr_size, lr_size]: 归一化后的低分辨率帧张量。
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        # decord 返回 Tensor[T, H, W, C]，值域 [0, 255]
        frames = vr.get_batch(frame_indices)  # [T, H, W, C]
        # 转为 float，归一化到 [0, 1]
        frames = frames.float() / 255.0  # [T, H, W, C]
        # 转换为 [T, C, H, W]
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        # 逐帧做 resize + normalize
        resized = []
        for t in range(frames.shape[0]):
            resized.append(self.lr_transform(frames[t]))  # [C, lr_size, lr_size]
        # 堆叠为 [C, T, lr_size, lr_size]
        lr_frames = torch.stack(resized, dim=1)
        return lr_frames

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.video_infos)

    def __getitem__(self, idx: int) -> dict:
        """返回单个样本字典。

        Returns:
            dict with keys:
                "lr_frames"     : Tensor[C, T, 224, 224]  低分辨率帧（已归一化）
                "video_path"    : str                      原始视频绝对路径
                "frame_indices" : List[int]                已采样帧的索引
                "mos"           : float                    主观质量分数
        """
        info = self.video_infos[idx]
        video_path = info["filename"]
        mos = info["mos"]

        # 获取视频总帧数
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        del vr  # 及时释放句柄

        # 采样帧索引
        frame_indices = self._sample_frame_indices(total_frames)

        # 加载低分辨率帧
        lr_frames = self._load_lr_frames(video_path, frame_indices)

        return {
            "lr_frames": lr_frames,           # Tensor[C, T, 224, 224]
            "video_path": video_path,          # 原始视频路径（CPU 侧使用）
            "frame_indices": frame_indices,    # List[int]
            "mos": torch.tensor(mos, dtype=torch.float32),
        }
