# 当前分支的运行环境

验证日期：2026-11-11（北京时间）。源码基线：`Effi_Base` / `614bc3b`。

已安装环境：`/home/gyf/newtest/ChangeFormer/.venv-facd`。

| 组件 | 已验证版本 |
| --- | --- |
| Python | 3.12.7 |
| PyTorch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| CUDA runtime / NVIDIA 驱动 | 12.1 / 550.107.02 |
| GPU | 2 × RTX 3090，24 GiB/卡 |
| timm / einops | 1.0.22 / 0.8.1 |
| NumPy / SciPy | 1.26.4 / 1.14.1 |

`requirements.txt` 固定主要依赖；`requirements.lock.txt` 保存本次环境全部 Python 包版本。
PyTorch 的 CUDA 运行库随 wheel 安装，本次验证没有单独安装 CUDA Toolkit、mmcv 或 torchaudio。
PyTorch 与 torchvision 的配对采用[官方安装组合](https://pytorch.org/get-started/previous-versions/#v251)。

## 使用与重建

```bash
source /home/gyf/newtest/ChangeFormer/.venv-facd/bin/activate
cd /home/gyf/newtest/ChangeFormer/test-project
python -m pip check
```

在另一台 Linux x86_64 机器上，先准备 Python 3.12 和可用的 NVIDIA 驱动，再创建独立环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
```

## 已完成验证

```bash
torchrun --standalone --nproc_per_node=1 scripts/validate_environment.py
torchrun --standalone --nproc_per_node=2 scripts/validate_environment.py
torchrun --standalone --nproc_per_node=1 main_cd.py --help
```

- 训练入口、评估模块导入成功；`pip check` 无依赖冲突。
- 验证脚本在临时目录生成 256×256 双时相图像、标签及列表，通过真实 `CDDataset` 执行读取、增强和标签归一化。
- 使用当前 `FreqFormer`、`CDTrainer` 和 AdamW，每卡 batch size 为 1，进行两次更新。
- 使用当前多尺度 CE 与 `0.1 × WaveletLoss`，没有修改概率处理或算法公式。
- 五级输出形状为 `[1,2,8,8]`、`[1,2,16,16]`、`[1,2,32,32]`、`[1,2,64,64]`、`[1,2,256,256]`。
- 单卡、双卡的输出、损失和梯度均有限，优化器实际更新参数；双卡检查了参数同步。
- `eval()` 前向与混淆矩阵指标计算成功。随机数据的指标没有准确率参考价值。
- 峰值 PyTorch 张量分配约 1708 MiB（单卡）和 1706 MiB/卡（双卡），不等同于全部 GPU 显存占用。
- 机器可读结果保存在 `environment-validation.json`。

本次验证没有运行完整 epoch、检查点保存恢复、训练完成后的自动测试流程，也没有验证所有对比模型。
尚未准备 LEVIR 数据，因此未验证实际数据训练、收敛及论文指标。正式训练前需要修改 `data_config.py` 中的数据路径和启动脚本的输出路径；模型参数使用 `--net_G FreqFormer`。

当前存在不阻断验证的警告：旧 `timm.models.layers` 导入路径弃用，DDP `find_unused_parameters=True` 产生额外开销；`main_cd.py --help` 在初始化进程组后由 argparse 直接退出，产生未显式销毁进程组警告。
