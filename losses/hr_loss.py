"""
losses/hr_loss.py

HR-KVQ 损失函数模块，包含：

1. 回归损失 (loss_reg)：
   - 对最终聚合质量分 Q 与真实 MOS 计算 MSE 损失（可选切换为 L1）。

2. Scale-consistent Local Perception Constraint (SC-LPC)（创新损失项）：
   - 对同一 Patch 的原始尺度（256×256）和下采样尺度（128×128）分别经过
     LocalPatchEncoder 后得到 q_i^256 和 q_i^128，
   - 约束两者一致：L_sc = L1Loss(q_i^256, q_i^128)
   - 直觉：4K 真实失真在多尺度下应保持可见 → 迫使模型学习尺度不变的失真感知。

总损失：
    L = L_reg + lambda_sc * L_sc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HRKVQLoss(nn.Module):
    """HR-KVQ 综合损失函数。

    Args:
        lambda_sc (float): SC-LPC 损失的权重系数，默认 0.1。
        reg_loss_type (str): 回归损失类型，"mse"（默认）或 "l1"。
        mos_scale (float): MOS 归一化因子（部分数据集 MOS 在 [0,100]，
                           此时可设 mos_scale=100 将 MOS 归一化到 [0,1]），默认 1.0。
    """

    def __init__(
        self,
        lambda_sc: float = 0.1,
        reg_loss_type: str = "mse",
        mos_scale: float = 1.0,
    ):
        super().__init__()
        self.lambda_sc = lambda_sc
        self.reg_loss_type = reg_loss_type
        self.mos_scale = mos_scale

        if reg_loss_type == "mse":
            self.reg_criterion = nn.MSELoss()
        elif reg_loss_type == "l1":
            self.reg_criterion = nn.L1Loss()
        else:
            raise ValueError(f"不支持的回归损失类型：{reg_loss_type}，请选择 'mse' 或 'l1'")

        self.sc_criterion = nn.L1Loss()

    # ------------------------------------------------------------------
    # 回归损失
    # ------------------------------------------------------------------

    def regression_loss(self, q_pred: torch.Tensor, mos: torch.Tensor) -> torch.Tensor:
        """计算预测质量分与真实 MOS 之间的回归损失。

        Args:
            q_pred (Tensor[B, 1]): 模型预测的视频质量分。
            mos    (Tensor[B] 或 Tensor[B, 1]): 真实主观质量分数。

        Returns:
            loss_reg (Tensor[scalar]): 回归损失值。
        """
        mos = mos.view_as(q_pred)
        if self.mos_scale != 1.0:
            mos = mos / self.mos_scale
        return self.reg_criterion(q_pred, mos)

    # ------------------------------------------------------------------
    # SC-LPC 损失
    # ------------------------------------------------------------------

    def sc_lpc_loss(
        self,
        q_256: torch.Tensor,
        q_128: torch.Tensor,
    ) -> torch.Tensor:
        """计算 Scale-consistent Local Perception Constraint 损失。

        Args:
            q_256 (Tensor[N, 1] 或 Tensor[B, P, 1]):
                原始 256×256 Patch 的 LocalPatchEncoder 输出质量分。
            q_128 (Tensor[N, 1] 或 Tensor[B, P, 1]):
                128×128 下采样 Patch 的 LocalPatchEncoder 输出质量分。

        Returns:
            loss_sc (Tensor[scalar]): 尺度一致性损失。
        """
        return self.sc_criterion(q_256, q_128)

    # ------------------------------------------------------------------
    # 综合前向
    # ------------------------------------------------------------------

    def forward(
        self,
        q_pred: torch.Tensor,
        mos: torch.Tensor,
        q_256: torch.Tensor = None,
        q_128: torch.Tensor = None,
    ) -> dict:
        """计算总损失（回归 + SC-LPC）。

        Args:
            q_pred (Tensor[B, 1]): 模型预测的视频质量分。
            mos    (Tensor[B] 或 Tensor[B, 1]): 真实 MOS。
            q_256  (Tensor, optional): 256 尺度 patch 质量分，启用 SC-LPC 时需提供。
            q_128  (Tensor, optional): 128 尺度 patch 质量分，启用 SC-LPC 时需提供。

        Returns:
            dict with keys:
                "total"   : 总损失（Tensor scalar）
                "reg"     : 回归损失（Tensor scalar）
                "sc_lpc"  : SC-LPC 损失（Tensor scalar，若未启用则为 0.0）
        """
        loss_reg = self.regression_loss(q_pred, mos)

        if q_256 is not None and q_128 is not None:
            loss_sc = self.sc_lpc_loss(q_256, q_128)
        else:
            loss_sc = torch.tensor(0.0, device=q_pred.device)

        total = loss_reg + self.lambda_sc * loss_sc

        return {
            "total": total,
            "reg": loss_reg,
            "sc_lpc": loss_sc,
        }
