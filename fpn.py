# standalone_fpn.py
# 这是一个移除了 MMLab 依赖的、完全独立的 FPN 实现

import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    """
    一个完全独立的 FPN 实现，只依赖 PyTorch。
    移除了 MMLab 的 BaseModule 和 ConvModule 依赖。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level
        
        self.start_level = start_level
        
        # 1. 构建“横向连接”层 (Lateral Connections)
        #    作用：将输入的多尺度特征图统一到 out_channels
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            # 将 ConvModule 替换为 nn.Conv2d
            l_conv = nn.Conv2d(
                in_channels[i],
                out_channels,
                kernel_size=1) # 1x1 卷积
            self.lateral_convs.append(l_conv)

        # 2. 构建 FPN 卷积层
        #    作用：在融合后的特征图上进行 3x3 卷积，以消除上采样带来的混叠效应
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
             # 将 ConvModule 替换为 nn.Conv2d
            fpn_conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3, # 3x3 卷积
                padding=1)
            self.fpn_convs.append(fpn_conv)
            
        # 初始化权重 (这是一个好的实践)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs):
        """Forward function."""
        # 确保输入特征图的数量和 in_channels 列表长度一致
        assert len(inputs) == len(self.in_channels)

        # --- 步骤 1: 执行横向连接 ---
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # --- 步骤 2: 构建自顶向下的通路 ---
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg)

        # --- 步骤 3: 在融合后的特征上应用 FPN 卷积 ---
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(self.num_outs)
        ]
        
        return tuple(outs)