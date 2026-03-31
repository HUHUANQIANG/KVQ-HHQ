"""
train_hr_kvq.py

HR-KVQ 自定义训练主程序。

严格按照以下 5 步 CPU/GPU 数据流进行每个训练 Step：
    1. DataLoader (CPU) → lr_frames, video_path, frame_indices, mos
    2. Global Forward (GPU) → saliency_map S_lr
    3. Saliency Mapping & Patch Cropping (CPU) → hr_patches, s_i
    4. Local Forward (GPU) → q_i, c_i
    5. Fusion & Loss (GPU) → Q, loss, backward

设计原则：
    - 全程不将完整 4K Tensor 加载到 GPU，显存峰值仅来自 Patch Encoder。
    - SC-LPC 损失在同一 Step 内对 256 和 128 两个尺度分别做 Local Forward。

用法示例：
    python train_hr_kvq.py \\
        --anno_file ./labels/train_labels.txt \\
        --data_prefix /data/4k_videos \\
        --epochs 40 \\
        --batch_size 2 \\
        --lr 5e-4
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

from datasets.vqa4k_dataset import VQA4KDataset
from models.hr_kvq import HR_KVQ
from losses.hr_loss import HRKVQLoss
from utils.patch_sampler import get_saliency_guided_patches


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """固定随机种子以保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(args, phase: str) -> DataLoader:
    """构建 VQA4KDataset 的 DataLoader。

    Args:
        args: 命令行参数对象。
        phase: "train" 或 "test"。

    Returns:
        DataLoader 实例。
    """
    dataset = VQA4KDataset(
        anno_file=args.anno_file if phase == "train" else args.val_anno_file,
        data_prefix=args.data_prefix,
        clip_len=args.clip_len,
        frame_interval=args.frame_interval,
        lr_size=args.lr_size,
        phase=phase,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(phase == "train"),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=(phase == "train"),
        # 注意：由于 frame_indices 是变长 list，需要自定义 collate，
        # 但此处各 sample 帧数相同（由 clip_len 固定），默认 collate 可用。
        collate_fn=_collate_fn,
    )
    return loader


def _collate_fn(batch: list) -> dict:
    """自定义 collate 函数，处理 frame_indices（List[int]）和字符串字段。

    Args:
        batch: List[dict]，每个 dict 包含 lr_frames, video_path, frame_indices, mos。

    Returns:
        collated dict.
    """
    lr_frames = torch.stack([b["lr_frames"] for b in batch], dim=0)
    video_paths = [b["video_path"] for b in batch]
    frame_indices = [b["frame_indices"] for b in batch]
    mos = torch.stack([b["mos"] for b in batch], dim=0)
    return {
        "lr_frames": lr_frames,
        "video_path": video_paths,
        "frame_indices": frame_indices,
        "mos": mos,
    }


class AverageMeter:
    """追踪滑动平均值的计数器。"""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: avg={self.avg:.4f}"


# ---------------------------------------------------------------------------
# 训练 Step（核心 CPU/GPU 数据流）
# ---------------------------------------------------------------------------

def train_step(
    model: HR_KVQ,
    criterion: HRKVQLoss,
    optimizer: torch.optim.Optimizer,
    data: dict,
    device: torch.device,
    patch_size: int = 256,
    top_k: int = 6,
    rand_k: int = 2,
    orig_h: int = 2160,
    orig_w: int = 3840,
    use_sc_lpc: bool = True,
) -> dict:
    """单次训练迭代，严格遵守 CPU/GPU 切换策略。

    Args:
        model:       HR_KVQ 模型（已位于 device）。
        criterion:   HRKVQLoss 损失函数。
        optimizer:   优化器。
        data:        DataLoader 返回的 batch 字典。
        device:      目标 GPU 设备。
        patch_size:  HR Patch 边长，默认 256。
        top_k:       Saliency 导引 patch 数，默认 6。
        rand_k:      随机补充 patch 数，默认 2。
        orig_h:      原始视频高度，默认 2160。
        orig_w:      原始视频宽度，默认 3840。
        use_sc_lpc:  是否启用 SC-LPC 尺度一致性损失，默认 True。

    Returns:
        dict: 各损失项的数值（已 detach 为 Python float）。
    """
    model.train()
    optimizer.zero_grad()

    # ------------------------------------------------------------------
    # Step 1：从 DataLoader batch 取出数据（均在 CPU）
    # ------------------------------------------------------------------
    lr_frames = data["lr_frames"]        # Tensor[B, C, T, 224, 224]，CPU
    video_paths = data["video_path"]     # List[str]
    frame_indices = data["frame_indices"]  # List[List[int]]
    mos = data["mos"]                    # Tensor[B]，CPU

    # ------------------------------------------------------------------
    # Step 2：Global Forward（GPU）→ Saliency Map S_lr
    # ------------------------------------------------------------------
    lr_frames_gpu = lr_frames.to(device, non_blocking=True)
    saliency_map = model.forward_global(lr_frames_gpu)   # Tensor[B, 1, H_s, W_s]，GPU

    # ------------------------------------------------------------------
    # Step 3：Saliency Mapping & Patch Cropping（CPU）
    #   - 将 saliency_map detach 并移回 CPU
    #   - 调用 patch_sampler 映射坐标并从原始视频裁剪 HR Patches
    # ------------------------------------------------------------------
    saliency_map_cpu = saliency_map.detach().cpu()   # 脱离计算图，移回 CPU

    hr_patches, saliency_weights = get_saliency_guided_patches(
        saliency_map_lr=saliency_map_cpu,
        video_path=video_paths,
        frame_indices=frame_indices,
        orig_h=orig_h,
        orig_w=orig_w,
        patch_size=patch_size,
        top_k=top_k,
        rand_k=rand_k,
    )
    # hr_patches:       Tensor[B, T, num_patches, C, patch_size, patch_size]，CPU
    # saliency_weights: Tensor[B, num_patches]，CPU

    # ------------------------------------------------------------------
    # Step 4：Local Forward（GPU）→ q_i, c_i
    # ------------------------------------------------------------------
    hr_patches_gpu = hr_patches.to(device, non_blocking=True)
    s_i_gpu = saliency_weights.to(device, non_blocking=True)  # [B, num_patches]
    mos_gpu = mos.to(device, non_blocking=True)

    q_i, c_i = model.forward_local(hr_patches_gpu)  # [B, num_patches, 1]

    # SC-LPC：同一区域下采样一半尺度版本也过 LocalPatchEncoder
    if use_sc_lpc:
        B, T, P, C_patch, H_p, W_p = hr_patches_gpu.shape
        # 下采样到原始 patch 尺寸的一半（在 GPU 上做 interpolate，节省 CPU→GPU 搬运）
        hr_patches_128 = F.interpolate(
            hr_patches_gpu.reshape(B * T * P, C_patch, H_p, W_p),
            size=(H_p // 2, W_p // 2),
            mode="bilinear",
            align_corners=False,
        ).view(B, T, P, C_patch, H_p // 2, W_p // 2)

        q_i_128, _ = model.forward_local(hr_patches_128)  # [B, P, 1]
    else:
        q_i_128 = None

    # ------------------------------------------------------------------
    # Step 5：Fusion & Loss（GPU）
    # ------------------------------------------------------------------
    Q, w_i = model.forward_fusion(q_i, c_i, s_i_gpu)  # Q: [B, 1]

    loss_dict = criterion(
        q_pred=Q,
        mos=mos_gpu,
        q_256=q_i if use_sc_lpc else None,
        q_128=q_i_128 if use_sc_lpc else None,
    )
    total_loss = loss_dict["total"]

    # Backward
    total_loss.backward()
    # 梯度裁剪，防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    return {
        "total": total_loss.item(),
        "reg": loss_dict["reg"].item(),
        "sc_lpc": loss_dict["sc_lpc"].item(),
    }


# ---------------------------------------------------------------------------
# 评估 Epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_epoch(
    model: HR_KVQ,
    loader: DataLoader,
    device: torch.device,
    patch_size: int = 256,
    top_k: int = 6,
    rand_k: int = 2,
    orig_h: int = 2160,
    orig_w: int = 3840,
) -> dict:
    """在验证集上评估 SRCC 和 PLCC。

    Args:
        model:   HR_KVQ 模型。
        loader:  验证集 DataLoader。
        device:  目标 GPU 设备。
        其余参数同 train_step。

    Returns:
        dict with keys "srcc" and "plcc".
    """
    model.eval()
    all_preds = []
    all_mos = []

    for data in tqdm(loader, desc="Evaluating"):
        lr_frames = data["lr_frames"].to(device, non_blocking=True)
        video_paths = data["video_path"]
        frame_indices = data["frame_indices"]
        mos = data["mos"]

        # Global forward
        saliency_map = model.forward_global(lr_frames)
        saliency_map_cpu = saliency_map.detach().cpu()

        # Patch sampling on CPU
        hr_patches, saliency_weights = get_saliency_guided_patches(
            saliency_map_lr=saliency_map_cpu,
            video_path=video_paths,
            frame_indices=frame_indices,
            orig_h=orig_h,
            orig_w=orig_w,
            patch_size=patch_size,
            top_k=top_k,
            rand_k=rand_k,
        )

        hr_patches_gpu = hr_patches.to(device, non_blocking=True)
        s_i_gpu = saliency_weights.to(device, non_blocking=True)

        # Local forward
        q_i, c_i = model.forward_local(hr_patches_gpu)

        # Fusion
        Q, _ = model.forward_fusion(q_i, c_i, s_i_gpu)

        all_preds.extend(Q.squeeze(-1).cpu().numpy().tolist())
        all_mos.extend(mos.numpy().tolist())

    all_preds = np.array(all_preds)
    all_mos = np.array(all_mos)

    srcc, _ = spearmanr(all_preds, all_mos)
    plcc, _ = pearsonr(all_preds, all_mos)
    return {"srcc": srcc, "plcc": plcc}


# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------

def train(args):
    """完整训练流程入口。

    三阶段训练策略（由 args.stage 控制）：
        Stage 1：仅训练 Global Saliency Branch（冻结 Local Encoder 和 Fusion Head）
        Stage 2：冻结 Global Branch，训练 Local Encoder 和 Fusion Head
        Stage 3：联合微调全部参数（小学习率）
    """
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # ------------------------------------------------------------------
    # 构建模型
    # ------------------------------------------------------------------
    model = HR_KVQ(
        pretrained_global=not args.no_pretrained_global,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_transformer_blocks=args.num_transformer_blocks,
        drop=args.drop,
        attn_drop=args.attn_drop,
    ).to(device)

    # ------------------------------------------------------------------
    # 阶段化参数冻结
    # ------------------------------------------------------------------
    if args.stage == 1:
        # Stage 1：只训练 Global Branch
        print("Stage 1：训练 Global Saliency Branch")
        for param in model.local_encoder.parameters():
            param.requires_grad = False
        for param in model.fusion_head.parameters():
            param.requires_grad = False
    elif args.stage == 2:
        # Stage 2：冻结 Global Branch，训练 Local + Fusion
        print("Stage 2：训练 Local Patch Encoder + Fusion Head")
        for param in model.global_branch.parameters():
            param.requires_grad = False
    else:
        # Stage 3：全部参数联合微调
        print("Stage 3：全参数联合微调")

    # 加载 checkpoint（若指定）
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"从 checkpoint 恢复：{args.resume}")

    # ------------------------------------------------------------------
    # 构建损失函数和优化器
    # ------------------------------------------------------------------
    criterion = HRKVQLoss(
        lambda_sc=args.lambda_sc,
        reg_loss_type=args.reg_loss_type,
        mos_scale=args.mos_scale,
    )

    # 不同模块使用不同学习率
    global_params = list(model.global_branch.parameters())
    local_params = list(model.local_encoder.parameters()) + list(model.fusion_head.parameters())

    param_groups = [
        {"params": [p for p in global_params if p.requires_grad], "lr": args.lr * args.backbone_lr_mult},
        {"params": [p for p in local_params if p.requires_grad], "lr": args.lr},
    ]
    # 过滤空 group
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)

    # ------------------------------------------------------------------
    # 构建 DataLoader
    # ------------------------------------------------------------------
    train_loader = build_dataloader(args, phase="train")
    has_val = bool(args.val_anno_file)
    if has_val:
        val_loader = build_dataloader(args, phase="test")

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    best_srcc = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}  (Stage {args.stage})")
        print(f"{'='*60}")

        meter_total = AverageMeter("total_loss")
        meter_reg = AverageMeter("reg_loss")
        meter_sc = AverageMeter("sc_lpc_loss")

        for step, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
            loss_dict = train_step(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                data=data,
                device=device,
                patch_size=args.patch_size,
                top_k=args.top_k,
                rand_k=args.rand_k,
                orig_h=args.orig_h,
                orig_w=args.orig_w,
                use_sc_lpc=not args.no_sc_lpc,
            )
            n = data["lr_frames"].shape[0]
            meter_total.update(loss_dict["total"], n)
            meter_reg.update(loss_dict["reg"], n)
            meter_sc.update(loss_dict["sc_lpc"], n)

            if (step + 1) % args.log_interval == 0:
                print(
                    f"  Step [{step+1}/{len(train_loader)}] "
                    f"total={meter_total.avg:.4f} "
                    f"reg={meter_reg.avg:.4f} "
                    f"sc_lpc={meter_sc.avg:.4f}"
                )

        scheduler.step()
        print(f"Epoch {epoch} 训练完成  {meter_total}")

        # 验证
        if has_val and (epoch % args.eval_interval == 0 or epoch == args.epochs):
            metrics = eval_epoch(
                model=model,
                loader=val_loader,
                device=device,
                patch_size=args.patch_size,
                top_k=args.top_k,
                rand_k=args.rand_k,
                orig_h=args.orig_h,
                orig_w=args.orig_w,
            )
            print(f"  验证结果 — SRCC: {metrics['srcc']:.4f}  PLCC: {metrics['plcc']:.4f}")

            if metrics["srcc"] > best_srcc:
                best_srcc = metrics["srcc"]
                ckpt_path = os.path.join(args.save_dir, f"best_stage{args.stage}.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": metrics,
                    },
                    ckpt_path,
                )
                print(f"  最优模型已保存至：{ckpt_path}  (SRCC={best_srcc:.4f})")

        # 定期保存 checkpoint
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"stage{args.stage}_epoch{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"  Checkpoint 保存至：{ckpt_path}")

    print(f"\n训练完成！Stage {args.stage} 最优 SRCC = {best_srcc:.4f}")


# ---------------------------------------------------------------------------
# 命令行参数解析
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="HR-KVQ 训练脚本")

    # 数据相关
    parser.add_argument("--anno_file", type=str, required=True, help="训练集标注文件路径")
    parser.add_argument("--val_anno_file", type=str, default="", help="验证集标注文件路径（可选）")
    parser.add_argument("--data_prefix", type=str, required=True, help="视频文件根目录")
    parser.add_argument("--clip_len", type=int, default=8, help="每视频采样帧数")
    parser.add_argument("--frame_interval", type=int, default=2, help="帧采样间隔")
    parser.add_argument("--lr_size", type=int, default=224, help="低分辨率帧边长")
    parser.add_argument("--orig_h", type=int, default=2160, help="原始视频高度")
    parser.add_argument("--orig_w", type=int, default=3840, help="原始视频宽度")

    # Patch 采样相关
    parser.add_argument("--patch_size", type=int, default=256, help="HR Patch 边长")
    parser.add_argument("--top_k", type=int, default=6, help="Top-K Saliency Patch 数")
    parser.add_argument("--rand_k", type=int, default=2, help="随机补充 Patch 数")

    # 模型结构
    parser.add_argument("--embed_dim", type=int, default=256, help="LocalPatchEncoder 特征维度")
    parser.add_argument("--num_heads", type=int, default=4, help="Transformer 注意力头数")
    parser.add_argument("--num_transformer_blocks", type=int, default=4, help="Transformer Block 数")
    parser.add_argument("--drop", type=float, default=0.1, help="Dropout 比例")
    parser.add_argument("--attn_drop", type=float, default=0.0, help="Attention Dropout 比例")
    parser.add_argument("--no_pretrained_global", action="store_true", default=False,
                        help="禁用 ImageNet 预训练的 GlobalSaliencyBranch（默认启用预训练）")

    # 损失函数
    parser.add_argument("--reg_loss_type", type=str, default="mse", choices=["mse", "l1"],
                        help="回归损失类型")
    parser.add_argument("--lambda_sc", type=float, default=0.1, help="SC-LPC 损失权重")
    parser.add_argument("--mos_scale", type=float, default=1.0,
                        help="MOS 归一化因子（如 MOS 在 [0,100] 则设为 100）")
    parser.add_argument("--no_sc_lpc", action="store_true", default=False,
                        help="禁用 SC-LPC 尺度一致性损失（默认启用）")

    # 训练配置
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="训练阶段：1=Global only, 2=Local+Fusion, 3=joint finetune")
    parser.add_argument("--epochs", type=int, default=40, help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--lr", type=float, default=5e-4, help="基础学习率")
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1,
                        help="Global Branch 学习率倍率（相对于基础学习率）")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="AdamW 权重衰减")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 工作进程数")
    parser.add_argument("--gpu", type=int, default=0, help="使用的 GPU 编号")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", type=str, default="", help="从指定 checkpoint 恢复训练")

    # 日志与保存
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="模型保存目录")
    parser.add_argument("--log_interval", type=int, default=20, help="日志打印间隔（step 数）")
    parser.add_argument("--eval_interval", type=int, default=5, help="验证间隔（epoch 数）")
    parser.add_argument("--save_interval", type=int, default=10, help="定期保存间隔（epoch 数）")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
