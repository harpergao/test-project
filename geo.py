import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

def find_chinese_font():
    """
    智能查找系统中可用的中文字体。
    """
    # 常见的中文字体名称列表
    font_names = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Songti SC', 'STHeiti', 'WenQuanYi Micro Hei']
    try:
        for font_name in font_names:
            try:
                # 尝试找到并返回第一个可用的字体
                return FontProperties(fname=plt.font_manager.findfont(FontProperties(family=font_name)))
            except Exception:
                continue
    except Exception:
        print("警告：未在系统中找到任何预期的中文字体。图表中的中文可能无法正常显示。")
        return None

def parse_log_file(file_path):
    """
    从指定的日志文件中读取内容并解析。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            log_text = f.read()
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'。请检查文件路径是否正确。")
        return pd.DataFrame()
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return pd.DataFrame()

    lr_pattern = re.compile(r"lr: ([\d.e-]+)")
    train_pattern = re.compile(r"Is_training: True\. \[(\d+),(\d+)\]\[([\d,]+),(\d+)\],.*?G_loss: ([\d.]+), running_mf1: ([\d.]+)")
    train_epoch_pattern = re.compile(r"Is_training: True\. Epoch (\d+) / \d+, epoch_mF1= ([\d.]+)")
    eval_epoch_pattern = re.compile(r"Is_training: False\. Epoch (\d+) / \d+, epoch_mF1= ([\d.]+)")
    eval_metrics_pattern = re.compile(r"acc: ([\d.]+) miou: ([\d.]+) mf1: ([\d.]+) iou_0: ([\d.]+) iou_1: ([\d.]+) F1_1: ([\d.]+)")

    records = []
    current_lr = None
    
    lines = log_text.strip().split('\n')
    for i, line in enumerate(lines):
        lr_match = lr_pattern.match(line)
        if lr_match:
            current_lr = float(lr_match.group(1))
        
        train_match = train_pattern.match(line)
        if train_match:
            epoch, _, step, _, g_loss, running_mf1 = train_match.groups()
            records.append({
                'type': 'train_step', 'epoch': int(epoch),
                'step': int(step.replace(',', '')), 'G_loss': float(g_loss),
                'running_mf1': float(running_mf1), 'lr': current_lr
            })

        train_epoch_match = train_epoch_pattern.match(line)
        if train_epoch_match:
            epoch, epoch_mf1 = train_epoch_match.groups()
            records.append({
                'type': 'train_epoch', 'epoch': int(epoch),
                'epoch_mf1': float(epoch_mf1), 'lr': current_lr
            })

        eval_epoch_match = eval_epoch_pattern.match(line)
        if eval_epoch_match:
            epoch, epoch_mf1 = eval_epoch_match.groups()
            if i + 1 < len(lines):
                metrics_match = eval_metrics_pattern.match(lines[i+1])
                if metrics_match:
                    acc, miou, mf1, iou_0, iou_1, f1_1 = metrics_match.groups()
                    records.append({
                        'type': 'eval_epoch', 'epoch': int(epoch),
                        'epoch_mf1': float(epoch_mf1), 'acc': float(acc), 'miou': float(miou),
                        'iou_1': float(iou_1), 'f1_1': float(f1_1), 'lr': current_lr
                    })
    return pd.DataFrame(records)

def visualize_training(df, font_prop):
    """
    使用解析后的DataFrame生成可视化图表。
    """
    if df.empty:
        print("未能从日志中解析出任何数据，无法生成图表。")
        return

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 定义通用文本属性
    title_kwargs = {'fontproperties': font_prop, 'fontsize': 16} if font_prop else {'fontsize': 16}
    label_kwargs = {'fontproperties': font_prop, 'fontsize': 12} if font_prop else {'fontsize': 12}
    
    fig.suptitle('训练过程可视化分析', **title_kwargs)

    train_step_df = df[df['type'] == 'train_step'].dropna(subset=['G_loss'])
    train_epoch_df = df[df['type'] == 'train_epoch'].dropna(subset=['epoch_mf1'])
    eval_epoch_df = df[df['type'] == 'eval_epoch'].dropna(subset=['epoch_mf1'])

    # 1. 训练损失 (G_loss)
    ax = axes[0, 0]
    if not train_step_df.empty:
        sns.lineplot(data=train_step_df, x='epoch', y='G_loss', ax=ax, label='训练损失 (G_loss)', errorbar=None)
        ax.set_title('每个Epoch内的训练损失变化', **title_kwargs)
        ax.set_xlabel('Epoch', **label_kwargs)
        ax.set_ylabel('G_loss', **label_kwargs)
        ax.legend()

    # 2. 训练与验证的mF1分数对比
    ax = axes[0, 1]
    if not train_epoch_df.empty:
        sns.lineplot(data=train_epoch_df, x='epoch', y='epoch_mf1', ax=ax, marker='o', linestyle='--', label='训练集 mF1')
    if not eval_epoch_df.empty:
        sns.lineplot(data=eval_epoch_df, x='epoch', y='epoch_mf1', ax=ax, marker='o', linestyle='-', label='验证集 mF1')
    if not train_epoch_df.empty or not eval_epoch_df.empty:
        ax.set_title('训练集 vs 验证集 mF1 分数', **title_kwargs)
        ax.set_xlabel('Epoch', **label_kwargs)
        ax.set_ylabel('mF1 Score', **label_kwargs)
        ax.legend()

    # 3. 学习率变化
    ax = axes[1, 0]
    lr_df = df.dropna(subset=['lr', 'epoch']).groupby('epoch')['lr'].first().reset_index()
    if not lr_df.empty:
        sns.lineplot(data=lr_df, x='epoch', y='lr', ax=ax, marker='.', label='学习率 (lr)')
        ax.set_title('学习率变化曲线', **title_kwargs)
        ax.set_xlabel('Epoch', **label_kwargs)
        ax.set_ylabel('Learning Rate', **label_kwargs)
        ax.legend()

    # 4. 验证集上关键指标
    ax = axes[1, 1]
    if not eval_epoch_df.empty:
        sns.lineplot(data=eval_epoch_df, x='epoch', y='iou_1', ax=ax, marker='s', label='变化类 IoU (iou_1)')
        sns.lineplot(data=eval_epoch_df, x='epoch', y='f1_1', ax=ax, marker='^', label='变化类 F1 (F1_1)')
        ax.set_title('验证集上变化类的关键指标', **title_kwargs)
        ax.set_xlabel('Epoch', **label_kwargs)
        ax.set_ylabel('Score', **label_kwargs)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="从日志文件可视化训练过程。")
    parser.add_argument('filepath', type=str, help='要解析的训练日志文件的路径。')
    args = parser.parse_args()
    
    print(f"正在读取日志文件: {args.filepath}")
    log_df = parse_log_file(args.filepath)
    
    print("\n解析后的数据概览:")
    print(log_df.head().to_string())
    
    # 查找中文字体
    chinese_font = find_chinese_font()
    
    visualize_training(log_df, chinese_font)

if __name__ == '__main__':
    main()