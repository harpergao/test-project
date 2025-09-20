import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'          # 国内镜像1
# os.environ['HF_ENDPOINT'] = 'https://modelscope.cn/'          # 国内镜像2


import torch
import timm

from fpn import FPN
from models.resnet import *

def main():
    # --- 1. 实例化骨干网络 (Backbone) ---
    # 使用 timm 库创建一个 EfficientNet-B5 模型
    # pretrained=True: 加载在 ImageNet 上预训练的权重
    # features_only=True: 这是关键！它让模型不返回最后的分类结果，
    #                      而是返回一个包含多个中间层特征图的列表，这正是 FPN 需要的输入格式。
    print("正在创建 EfficientNet-B5 模型...")
    backbone = timm.create_model(
        'efficientnet_b4',
        pretrained=True,
        features_only=True,
    )
    # backbone = resnet18(pretrained=True,
    #                     replace_stride_with_dilation=[False,True,True])
    backbone.eval() # 设置为评估模式，关闭 Dropout 等
    print("模型创建完成。\n")
    info = backbone.feature_info

    print("EfficientNet-B5 特征提取信息:")
    print("-" * 40)
    for i, feature_info in enumerate(info):
        print(f"特征层索引 {i}:")
        print(f"  - 通道数 (Channels): {feature_info['num_chs']}")
        print(f"  - 步长/缩减率 (Stride): {feature_info['reduction']}")
        print(f"  - 来源模块 (Module): {feature_info['module']}")
    print("-" * 40)
    # --- 2. 准备一个模拟输入 ---
    # 创建一个随机的输入张量 (Tensor) 来模拟一张图片
    # 尺寸: (batch_size, channels, height, width)
    # 这里模拟一个 batch 为 1，3通道（RGB），尺寸为 512x512 的图片
    dummy_input = torch.randn(1, 3, 256, 256)
    print(f"创建模拟输入，尺寸: {dummy_input.shape}\n")

    # --- 3. 获取骨干网络的输出特征 ---
    # 将模拟输入送入骨干网络，获取输出的特征图列表
    # 使用 torch.no_grad() 来确保在前向传播时不会计算梯度，节省计算资源
    with torch.no_grad():
        backbone_features = backbone(dummy_input)

    # 动态获取 FPN 需要的 in_channels 参数
    # backbone_features 是一个列表，其中每个元素都是一个特征图
    # 我们遍历这个列表，获取每个特征图的通道数 (shape[1])
    in_channels = [f.shape[1] for f in backbone_features]
    
    print("--- 骨干网络输出信息 ---")
    print(f"输出了 {len(backbone_features)} 个层级的特征图")
    print(f"各层级特征图的通道数 (in_channels for FPN): {in_channels}")
    for i, features in enumerate(backbone_features):
        print(f"  特征层 {i+1} 的尺寸: {features.shape}")
    print("-" * 25 + "\n")

    # --- 4. 实例化 FPN (Neck) ---
    # 现在我们有了 FPN 所需的所有关键参数
    print("正在根据骨干网络的输出配置并创建 FPN...")
    fpn = FPN(
        in_channels=in_channels, # 刚刚动态获取的通道数列表
        out_channels=256,       # FPN 输出的所有特征图的统一通道数，256 是一个常用值
        num_outs=len(in_channels) # FPN 输出的特征图数量，通常和输入数量一致
    )
    print("FPN 创建完成。\n")


    # --- 5. 将骨干网络的输出送入 FPN ---
    # 这就是“调用”的核心步骤
    print("正在将骨干网络的特征送入 FPN...")
    fpn_outputs = fpn(backbone_features)


    # --- 6. 查看 FPN 的输出 ---
    print("\n--- FPN 输出信息 ---")
    print(f"FPN 输出了 {len(fpn_outputs)} 个层级的特征图")
    print(f"所有输出特征图的通道数都已统一为: {fpn.out_channels}")
    for i, output in enumerate(fpn_outputs):
        print(f"  输出层 {i+1} 的尺寸: {output.shape}")
    print("-" * 25)

if __name__ == '__main__':
    main()