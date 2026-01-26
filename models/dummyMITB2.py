import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'          # 国内镜像1
from transformers import SegformerModel, SegformerConfig

# 1. 加载预训练的 MiT-B2 Backbone (去掉分类头)
# "nvidia/mit-b2" 会自动加载 ImageNet-1k 的预训练权重
model = SegformerModel.from_pretrained("nvidia/mit-b2")

# 2. 模拟一个输入 (Batch=2, Channel=3, 256x256)
# 注意：真实训练时，这里就是你的 DataLoader 出来的 tensor
dummy_input = torch.randn(2, 3, 256, 256)

# 3. 前向传播
# 关键点：设置 output_hidden_states=True 以获取每一层的特征
outputs = model(dummy_input, output_hidden_states=True)

# 4. 提取多尺度特征
# outputs.hidden_states 是一个元组，通常包含：(Embedding输出, Stage1输出, Stage2输出, Stage3输出, Stage4输出)
# 我们通常需要最后 4 个，分别对应 1/4, 1/8, 1/16, 1/32

features = outputs.hidden_states

print("Input shape:", dummy_input.shape)
print("-" * 30)

# 注意：SegFormer 的 hidden_states 可能包含 Input Embedding 的输出
# 通常我们要的是索引为 1, 2, 3, 4 的特征（具体视版本而定，建议打印 shape 确认）
# 或者是直接取最后4个：features[-4], features[-3], features[-2], features[-1]

stage_1 = features[-4] # 1/4
stage_2 = features[-3] # 1/8
stage_3 = features[-2] # 1/16
stage_4 = features[-1] # 1/32

print(f"Stage 1 (1/4):  {stage_1.shape}") # 预期: [2, 64, 64, 64]
print(f"Stage 2 (1/8):  {stage_2.shape}") # 预期: [2, 128, 32, 32]
print(f"Stage 3 (1/16): {stage_3.shape}") # 预期: [2, 320, 16, 16]
print(f"Stage 4 (1/32): {stage_4.shape}") # 预期: [2, 512, 8, 8]