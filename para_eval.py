import torch
import torch.nn as nn
from thop import profile
from types import SimpleNamespace # 用于创建 args 对象

from models.trainer import *




# 为其他模型也创建简单占位符
ChangeFormerV6 = SiamUnet_diff = ResNet 
# ... 您可以为所有模型创建类似的占位符，或者直接从您的文件中导入

# ==============================================================================
# --- 主程序入口 ---
# ==============================================================================
if __name__ == "__main__":
    # --- 步骤 1: 设置要测试的模型 ---
    args = SimpleNamespace()
    args.net_G = 'base_efficientnet_b4' # 您可以换成任何想测试的模型
    args.embed_dim = 256 # 确保为需要它的模型提供参数

    # --- 步骤 1.5: 定义您要使用的设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # --- 步骤 2: 定义模型 ---
    # 正常调用您的函数，它可能会返回一个被 DataParallel 包装过的模型
    model_wrapped = define_G(args, gpu_ids=[0] if str(device) == 'cuda' else [])

    # --- 步骤 2.5: 解开 DataParallel 包装！ ---
    # 这是修复错误的关键步骤
    if isinstance(model_wrapped, torch.nn.DataParallel):
        model = model_wrapped.module
        print("INFO: 检测到 DataParallel 包装，已提取原始模型进行分析。")
    else:
        model = model_wrapped

    # 确保最终的模型在正确的设备上并处于评估模式
    model.to(device)
    model.eval()

    # --- 步骤 3: 创建虚拟张量并移动到设备 ---
    input_height = 256
    input_width = 256
    
    # 创建一个在CPU上的元组
    dummy_input_cpu = (torch.randn(1, 3, input_height, input_width),
                       torch.randn(1, 3, input_height, input_width))
    
    # 将元组中的每个张量移动到目标设备
    dummy_input = tuple(inp.to(device) for inp in dummy_input_cpu)
    
    print(f"使用的虚拟输入尺寸: (1, 3, {input_height}, {input_width}) x 2")

    # --- 步骤 4: 运行 thop 进行计算 ---
    try:
        # 现在传递给 profile 的是原始模型，它和输入都在同一个设备上
        macs, params = profile(model, inputs=dummy_input, verbose=False)
        
        # --- 步骤 5: 打印结果 ---
        print("-" * 40)
        print(f"模型 分析结果:")
        # 将参数量转换为 "M" (百万) 为单位
        print(f"  参数量 (Params) : {params / 1e6:.4f} M")
        # 将MACs转换为 "G" (十亿) 为单位。FLOPs 约等于 2 * MACs
        print(f"  计算量 (MACs)   : {macs / 1e9:.4f} G")
        print(f"  估算FLOPs       : {(macs * 2) / 1e9:.4f} G")
        print("-" * 40)

    except Exception as e:
        print(f"\n计算过程中发生错误:\n{e}")
