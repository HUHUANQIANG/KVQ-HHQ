"""
models/hr_kvq.py

HR-KVQ（High-Resolution KVQ）模型：面向 4K 超高清视频质量评估的双路网络。

核心模块：
    GlobalSaliencyBranch  — 低分辨率（224×224）输入，输出 Saliency Map S_lr
    LocalPatchEncoder     — 高分辨率 256×256 Patch 输入，输出 patch 质量分 q_i 和置信度 c_i
    FusionHead            — 基于 s_i / c_i / q_i 加权聚合得到最终视频质量分 Q
    HR_KVQ                — 完整模型（组装以上模块）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from timm.models.layers import trunc_normal_, DropPath

# ---------------------------------------------------------------------------
# 工具模块
# ---------------------------------------------------------------------------

class _Mlp(nn.Module):
    """标准 Transformer MLP Block（两层线性 + GELU + Dropout）。"""

    def __init__(self, in_features: int, hidden_features: int = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class _SelfAttention(nn.Module):
    """标准多头自注意力模块。"""

    def __init__(self, dim: int, num_heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class _TransformerBlock(nn.Module):
    """标准 Transformer Block：Pre-Norm + MSA + FFN。"""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SelfAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# GlobalSaliencyBranch
# ---------------------------------------------------------------------------

class GlobalSaliencyBranch(nn.Module):
    """低分辨率（224×224）全局 Saliency 估计分支。

    复用 ResNet-50 作为特征提取主干，附加轻量 Conv Head 输出单通道 Saliency Map。
    若不需要 Saliency Map 的绝对空间精度，可将 backbone 替换为更轻量的模型
    （如 MobileNetV3 或 EfficientNet-B0）。

    Forward Input:
        x (Tensor[B, C, T, H, W]): B 个视频片段，每帧已下采样至 224×224。
                                    C=3, T=帧数, H=W=224。
    Forward Output:
        saliency_map (Tensor[B, 1, H_s, W_s]): 低分辨率 Saliency Map，
            空间尺寸由 backbone stride 决定，约为 7×7 或 14×14（ResNet 默认）。
    """

    def __init__(
        self,
        pretrained: bool = True,
        saliency_channels: int = 64,
    ):
        super().__init__()

        # 使用 torchvision 提供的 ResNet-50 作为轻量 backbone
        import torchvision.models as tvm
        resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        # 去掉全局平均池化层和分类头，保留到 layer4（输出 [B, 2048, H/32, W/32]）
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        backbone_out_channels = 2048  # ResNet-50 layer4 输出通道数

        # 轻量 Saliency Head：将高维特征压缩为单通道响应图
        self.saliency_head = nn.Sequential(
            nn.Conv2d(backbone_out_channels, saliency_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(saliency_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(saliency_channels, 1, kernel_size=1, bias=True),
        )

        # 初始化 Saliency Head 权重
        for m in self.saliency_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor[B, C, T, H, W]

        Returns:
            saliency_map: Tensor[B, 1, H_s, W_s]，对时间维度取平均后的空间 Saliency。
        """
        B, C, T, H, W = x.shape
        # 将时间维度展开，按帧分别提取特征（ResNet 不含时间建模）
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T, C, H, W]
        feat = self.backbone(x_2d)           # [B*T, 2048, H_s, W_s]
        sal = self.saliency_head(feat)       # [B*T, 1, H_s, W_s]
        # 恢复时间维度并对帧取均值
        sal = sal.view(B, T, 1, sal.shape[2], sal.shape[3])  # [B, T, 1, H_s, W_s]
        saliency_map = sal.mean(dim=1)       # [B, 1, H_s, W_s]
        return saliency_map


# ---------------------------------------------------------------------------
# LocalPatchEncoder
# ---------------------------------------------------------------------------

class LocalPatchEncoder(nn.Module):
    """高分辨率 Patch 局部质量编码器。

    架构：
        CNN Stem（3 层 Conv2d 3×3 stride=2）
            → 捕捉 4K 特有的 block artifact / ringing / fine-texture blur
        Transformer（4 个标准 Self-Attention Block）
            → 建模局部 patch 内的结构与纹理关系
        双头输出：
            q_i  → patch 质量分（标量）
            c_i  → patch 置信度（标量，经 Sigmoid 映射到 (0,1)）

    Forward Input:
        patches (Tensor[N, C, patch_size, patch_size]):
            N 个 patch，C=3，默认 patch_size=256。
    Forward Output:
        q (Tensor[N, 1]): 每个 patch 的质量分。
        c (Tensor[N, 1]): 每个 patch 的置信度。
    """

    def __init__(
        self,
        patch_size: int = 256,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_transformer_blocks: int = 4,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size

        # ------------------------------------------------------------------
        # CNN Stem：3 层 Conv2d 3×3 stride=2，将 256→32 空间维度
        # 专门用于提取高频失真特征（block/ringing artifacts）
        # ------------------------------------------------------------------
        stem_channels = [3, 64, 128, embed_dim]
        stem_layers = []
        for i in range(3):
            in_ch = stem_channels[i]
            out_ch = stem_channels[i + 1]
            stem_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ]
        self.stem = nn.Sequential(*stem_layers)
        # 经过 3 次 stride=2 后，256 → 32（spatial tokens 数：32×32=1024）

        # 将 CNN 输出展平为 token 序列的投影层
        self.token_proj = nn.Linear(embed_dim, embed_dim)

        # 位置编码（可学习）
        spatial_size = patch_size // (2 ** 3)  # 3 层 stride=2 后的空间尺寸（patch_size / 8）
        num_tokens = spatial_size * spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

        # ------------------------------------------------------------------
        # Transformer Blocks
        # ------------------------------------------------------------------
        self.blocks = nn.ModuleList([
            _TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
            )
            for _ in range(num_transformer_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ------------------------------------------------------------------
        # 双输出头
        # ------------------------------------------------------------------
        self.q_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
        )
        self.c_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),  # 置信度归一化到 (0,1)
        )

    def forward(self, patches: torch.Tensor) -> tuple:
        """
        Args:
            patches: Tensor[N, C, patch_size, patch_size]

        Returns:
            q: Tensor[N, 1]  patch 质量分
            c: Tensor[N, 1]  patch 置信度
        """
        # CNN Stem
        feat = self.stem(patches)        # [N, embed_dim, H', W']
        N, C, H_, W_ = feat.shape
        # 展平为 token 序列
        tokens = feat.flatten(2).transpose(1, 2)  # [N, H'*W', embed_dim]
        tokens = self.token_proj(tokens)

        # 加位置编码
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]

        # Transformer Blocks
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # Global Average Pooling over tokens
        feat_pooled = tokens.mean(dim=1)  # [N, embed_dim]

        q = self.q_head(feat_pooled)  # [N, 1]
        c = self.c_head(feat_pooled)  # [N, 1]
        return q, c


# ---------------------------------------------------------------------------
# FusionHead
# ---------------------------------------------------------------------------

class FusionHead(nn.Module):
    """基于 Saliency 权重和置信度的加权聚合 Fusion Head。

    聚合公式：
        w_i = softmax(α * s_i + β * c_i)
        Q   = Σ (w_i * q_i)

    其中 α、β 为可学习标量参数。

    Forward Input:
        q_i (Tensor[B, num_patches, 1]): 每个 patch 的质量分。
        c_i (Tensor[B, num_patches, 1]): 每个 patch 的置信度。
        s_i (Tensor[B, num_patches]):    每个 patch 的 Saliency 权重（已归一化）。

    Forward Output:
        Q   (Tensor[B, 1]): 视频级别的聚合质量分。
        w_i (Tensor[B, num_patches]): 最终聚合权重（用于可视化/调试）。
    """

    def __init__(self, init_alpha: float = 1.0, init_beta: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(
        self,
        q_i: torch.Tensor,
        c_i: torch.Tensor,
        s_i: torch.Tensor,
    ) -> tuple:
        """
        Args:
            q_i: Tensor[B, num_patches, 1]
            c_i: Tensor[B, num_patches, 1]
            s_i: Tensor[B, num_patches]

        Returns:
            Q:   Tensor[B, 1]
            w_i: Tensor[B, num_patches]
        """
        # 对齐维度
        c_i_flat = c_i.squeeze(-1)   # [B, num_patches]
        q_i_flat = q_i.squeeze(-1)   # [B, num_patches]

        # 聚合权重：α·s + β·c，然后 softmax
        logits = self.alpha * s_i + self.beta * c_i_flat  # [B, num_patches]
        w_i = F.softmax(logits, dim=-1)                   # [B, num_patches]

        # 加权求和
        Q = (w_i * q_i_flat).sum(dim=-1, keepdim=True)   # [B, 1]
        return Q, w_i


# ---------------------------------------------------------------------------
# HR_KVQ（主模型）
# ---------------------------------------------------------------------------

class HR_KVQ(nn.Module):
    """HR-KVQ 完整模型：双路 4K 视频质量评估网络。

    路径 A（Global Branch）：
        输入 224×224 低分辨率帧 → GlobalSaliencyBranch → Saliency Map S_lr

    路径 B（Local Branch）：
        输入 256×256 高分辨率 Patch → LocalPatchEncoder → (q_i, c_i)

    Fusion：
        FusionHead 结合 s_i（Saliency 权重）、c_i（置信度）、q_i（patch 质量）→ Q

    注意：模型的 forward 只负责 GPU 上的计算（Global forward 和 Local forward）；
    Saliency Map 到 Patch 的映射与裁剪（CPU 操作）由 train_hr_kvq.py 中的
    训练循环调用 patch_sampler.py 来完成。
    """

    def __init__(
        self,
        pretrained_global: bool = True,
        patch_size: int = 256,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_transformer_blocks: int = 4,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
        attn_drop: float = 0.0,
    ):
        super().__init__()

        # 全局低分辨率 Saliency 分支
        self.global_branch = GlobalSaliencyBranch(pretrained=pretrained_global)

        # 局部高分辨率 Patch 质量编码器
        self.local_encoder = LocalPatchEncoder(
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_transformer_blocks=num_transformer_blocks,
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
        )

        # 聚合 Fusion Head
        self.fusion_head = FusionHead()

    # ------------------------------------------------------------------
    # 阶段化 forward 接口（供训练循环分步调用）
    # ------------------------------------------------------------------

    def forward_global(self, lr_frames: torch.Tensor) -> torch.Tensor:
        """第一步（GPU）：用低分辨率帧得到 Saliency Map。

        Args:
            lr_frames: Tensor[B, C, T, 224, 224]

        Returns:
            saliency_map: Tensor[B, 1, H_s, W_s]
        """
        return self.global_branch(lr_frames)

    def forward_local(self, hr_patches: torch.Tensor) -> tuple:
        """第四步（GPU）：对 HR Patches 编码，输出 patch 质量分和置信度。

        Args:
            hr_patches: Tensor[B, T, num_patches, C, patch_size, patch_size]

        Returns:
            q_i: Tensor[B, num_patches, 1]  （对时间维度取均值后）
            c_i: Tensor[B, num_patches, 1]
        """
        B, T, P, C, H, W = hr_patches.shape
        # 展开为 (B*T*P, C, H, W) 批量处理
        patches_flat = hr_patches.reshape(B * T * P, C, H, W)
        q_flat, c_flat = self.local_encoder(patches_flat)  # [B*T*P, 1]
        # 恢复维度并对时间取均值
        q = q_flat.view(B, T, P, 1).mean(dim=1)  # [B, P, 1]
        c = c_flat.view(B, T, P, 1).mean(dim=1)  # [B, P, 1]
        return q, c

    def forward_fusion(
        self,
        q_i: torch.Tensor,
        c_i: torch.Tensor,
        s_i: torch.Tensor,
    ) -> tuple:
        """第五步（GPU）：聚合得到最终视频质量分 Q。

        Args:
            q_i: Tensor[B, num_patches, 1]
            c_i: Tensor[B, num_patches, 1]
            s_i: Tensor[B, num_patches]   Saliency 权重（来自 patch_sampler）

        Returns:
            Q:   Tensor[B, 1]
            w_i: Tensor[B, num_patches]
        """
        return self.fusion_head(q_i, c_i, s_i)

    # ------------------------------------------------------------------
    # 完整 end-to-end forward（适用于测试阶段，此时 hr_patches 和 s_i 已在外部准备好）
    # ------------------------------------------------------------------

    def forward(
        self,
        lr_frames: torch.Tensor,
        hr_patches: torch.Tensor,
        s_i: torch.Tensor,
    ) -> tuple:
        """端到端 forward（测试/推理阶段使用）。

        训练阶段建议使用分步接口（forward_global / forward_local / forward_fusion），
        以精确控制 CPU/GPU 数据流，避免全 4K Tensor 进入 GPU。

        Args:
            lr_frames:  Tensor[B, C, T, 224, 224]         低分辨率帧
            hr_patches: Tensor[B, T, num_patches, C, H, W] 高分辨率 Patches（已在 CPU 裁剪）
            s_i:        Tensor[B, num_patches]              Saliency 权重

        Returns:
            Q:           Tensor[B, 1]   视频质量分
            saliency:    Tensor[B, 1, H_s, W_s]  Saliency Map（可用于可视化）
            w_i:         Tensor[B, num_patches]  聚合权重
        """
        saliency = self.forward_global(lr_frames)
        q_i, c_i = self.forward_local(hr_patches)
        Q, w_i = self.forward_fusion(q_i, c_i, s_i)
        return Q, saliency, w_i
