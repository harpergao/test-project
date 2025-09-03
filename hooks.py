import torch
import torch.nn as nn
from types import SimpleNamespace

# 导入你的模型定义函数和具体的模型类
# 假设你的 networks.py 和 models 文件夹在 Python 的搜索路径中
from models.networks import define_G, ResNet
from models.ChangeFormer import ChangeFormerV1, ChangeFormerV2, ChangeFormerV3, ChangeFormerV4, ChangeFormerV5, ChangeFormerV6
from models.SiamUnet_diff import SiamUnet_diff
from models.help_funcs import Transformer, TransformerDecoder # 导入自定义模块以便识别

# ==============================================================================
# --- 核心：钩子函数 ---
# ==============================================================================
def get_shape_hook(name):
    """
    这是一个钩子生成器。它返回一个钩子函数，
    该函数会打印出其所注册的模块的输出形状。
    """
    def hook(module, input, output):
        # 为了美观，对模块名进行格式化输出
        # 有些模块（如自定义的Transformer）可能返回元组，需要分开处理
        if isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"[{name:<40}] - Output {i} Shape: {list(out.shape)}")
        elif isinstance(output, torch.Tensor):
            print(f"[{name:<40}] - Output Shape: {list(output.shape)}")
    return hook

# ==============================================================================
# --- 主程序入口 ---
# ==============================================================================
if __name__ == "__main__":
    # --- 步骤 1: 设置要测试的模型 ---
    args = SimpleNamespace()
    # ========================================================
    #  在这里修改你想测试的模型
    args.net_G = 'base_transformer_pos_s4_dd8' 
    # ========================================================
    args.embed_dim = 256 # 确保为需要它的模型提供参数

    # --- 步骤 1.5: 定义您要使用的设备 ---
    device = torch.device("cpu") # 调试时使用CPU更方便，避免GPU显存问题
    print(f"使用的设备: {device}")

    # --- 步骤 2: 定义模型 ---
    model_wrapped = define_G(args, gpu_ids=[]) # gpu_ids留空，在CPU上创建模型

    # 解开 DataParallel 包装（如果存在）
    if isinstance(model_wrapped, torch.nn.DataParallel):
        model = model_wrapped.module
    else:
        model = model_wrapped

    model.to(device)
    model.eval()

    # --- 步骤 3: 注册钩子 ---
    hook_handles = []
    print("\n" + "="*30 + " 注册钩子 " + "="*30)
    for name, layer in model.named_modules():
        # 你可以根据需要选择要监控的层类型
        # 这里我们选择了一些关键层来观察尺寸变化
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.Upsample, Transformer, TransformerDecoder)):
            handle = layer.register_forward_hook(get_shape_hook(name))
            hook_handles.append(handle)
            print(f"已为层 '{name}' ({layer.__class__.__name__}) 注册钩子")
    
    # --- 步骤 4: 创建虚拟张量并运行前向传播 ---
    input_height = 256
    input_width = 256
    
    dummy_input_t1 = torch.randn(1, 3, input_height, input_width).to(device)
    dummy_input_t2 = torch.randn(1, 3, input_height, input_width).to(device)
    
    print("\n" + "="*25 + " 开始前向传播并追踪尺寸 " + "="*25)
    print(f"输入尺寸: {list(dummy_input_t1.shape)}\n")
    
    with torch.no_grad(): # 在评估模式下，不需要计算梯度
        # 模型需要两个输入，所以我们分别传入
        final_output = model(dummy_input_t1, dummy_input_t2)

    # --- 步骤 5: 打印最终输出并移除钩子 ---
    print("\n" + "="*30 + " 传播完成 " + "="*30)
    
    # 模型的输出可能是一个列表或元组
    if isinstance(final_output, (tuple, list)):
        for i, out in enumerate(final_output):
             if isinstance(out, torch.Tensor):
                print(f"最终输出 {i} 的尺寸: {list(out.shape)}")
    elif isinstance(final_output, torch.Tensor):
        print(f"最终输出的尺寸: {list(final_output.shape)}")

    # 移除钩子，这是个好习惯，防止内存泄漏
    for handle in hook_handles:
        handle.remove()
    print("\n所有钩子已移除。")