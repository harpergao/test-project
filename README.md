A TEST:
From https://github.com/wgcban/ChangeFormer/

## 快速配置环境（Effi_Base）

已在 Python 3.12.7、PyTorch 2.5.1 + CUDA 12.1、双 RTX 3090 上验证。
另一台 Linux x86_64 服务器需要预先准备 Python 3.12 和兼容的 NVIDIA 驱动。

```bash
git clone --branch Effi_Base https://github.com/harpergao/test-project.git
cd test-project
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip check
torchrun --standalone --nproc_per_node=1 scripts/validate_environment.py
# 两张可用 GPU 时验证双卡：
torchrun --standalone --nproc_per_node=2 scripts/validate_environment.py
```

验证使用临时合成数据，不需要下载 LEVIR；通过不代表复现论文指标。
真实训练前仍需准备数据并修改路径。环境详情与验证范围见 [ENVIRONMENT.md](ENVIRONMENT.md)。
