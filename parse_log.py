import re
import csv
import os

def parse_log_file(log_file_path, output_csv_path):
    """
    解析深度学习训练日志，为每个 Epoch 生成一行包含训练和评估的完整数据。
    
    新规定列: 
    - Epoch, Avg_Training_Loss, Avg_Evaluation_Loss, 
    - Training_F1_1, Training_Precision_1, Training_Recall_1,
    - Evaluation_F1_1, Evaluation_Precision_1, Evaluation_Recall_1
    """
    print(f"🚀 开始处理日志文件: {log_file_path}")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ 错误：找不到日志文件 '{log_file_path}'。")
        return
    except Exception as e:
        print(f"❌ 错误：读取文件时发生意外错误。详细信息: {e}")
        return

    # 正则表达式
    # 捕获 Epoch 编号，作为启动一个新数据容器的信号
    epoch_start_pattern = re.compile(r"Is_training: True. \[(\d+),")
    train_loss_pattern = re.compile(r"Is_training: True.*G_loss: ([\d.]+)")
    eval_loss_pattern = re.compile(r"Is_training: False.*G_loss: ([\d.]+)")
    # 此模式用于匹配任何一个指标行 (训练或评估)
    metrics_pattern = re.compile(r"acc: .*?F1_1: ([\d.]+).*?precision_1: ([\d.]+).*?recall_1: ([\d.]+)")

    # --- 核心逻辑重构 ---
    all_epochs_data = []
    current_epoch_data = {}
    temp_train_losses = []
    temp_eval_losses = []

    for line in lines:
        epoch_match = epoch_start_pattern.search(line)
        # 当找到一个训练起始行时，我们判断是否需要开启一个新的 Epoch 记录
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            # 如果当前容器是空的，或者遇到了一个全新的epoch编号
            if not current_epoch_data or current_epoch_data.get('Epoch') != epoch_num:
                current_epoch_data = {'Epoch': epoch_num}
                temp_train_losses = []
                temp_eval_losses = []

        # 只要容器被初始化，就开始收集loss
        if current_epoch_data:
            train_loss_match = train_loss_pattern.search(line)
            if train_loss_match:
                temp_train_losses.append(float(train_loss_match.group(1)))

            eval_loss_match = eval_loss_pattern.search(line)
            if eval_loss_match:
                temp_eval_losses.append(float(eval_loss_match.group(1)))

            # 当匹配到任何一个指标行 (acc: ...)
            metrics_match = metrics_pattern.search(line)
            if metrics_match:
                f1_1 = float(metrics_match.group(1))
                precision_1 = float(metrics_match.group(2))
                recall_1 = float(metrics_match.group(3))

                # 判断这是训练指标还是评估指标
                if 'Training_F1_1' not in current_epoch_data:
                    # 如果容器里还没有训练F1值，说明这是第一个指标行，即训练指标
                    current_epoch_data['Avg_Training_Loss'] = sum(temp_train_losses) / len(temp_train_losses) if temp_train_losses else None
                    current_epoch_data['Training_F1_1'] = f1_1
                    current_epoch_data['Training_Precision_1'] = precision_1
                    current_epoch_data['Training_Recall_1'] = recall_1
                else:
                    # 如果容器里已有训练F1值，说明这是第二个指标行，即评估指标
                    current_epoch_data['Avg_Evaluation_Loss'] = sum(temp_eval_losses) / len(temp_eval_losses) if temp_eval_losses else None
                    current_epoch_data['Evaluation_F1_1'] = f1_1
                    current_epoch_data['Evaluation_Precision_1'] = precision_1
                    current_epoch_data['Evaluation_Recall_1'] = recall_1

                    # --- 一个Epoch的数据收集完整！---
                    all_epochs_data.append(current_epoch_data)
                    # 清空容器，为下一个epoch做准备
                    current_epoch_data = {}

    # --- 写入CSV文件 ---
    if not all_epochs_data:
        print("🤷‍ 未能从日志中提取到任何完整的 Epoch 数据。")
        return

    # 定义新的表头
    fieldnames = [
        'Epoch', 'Avg_Training_Loss', 'Avg_Evaluation_Loss', 
        'Training_F1_1', 'Training_Precision_1', 'Training_Recall_1',
        'Evaluation_F1_1', 'Evaluation_Precision_1', 'Evaluation_Recall_1'
    ]
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # 写入时，使用 get(key, None) 来避免因某个值缺失导致程序错误
        for row in all_epochs_data:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"✅ 数据提取完成！已合并并保存到文件: {output_csv_path}")
    print(f"共处理了 {len(all_epochs_data)} 个 Epoch 的数据。")


if __name__ == '__main__':
    input_log_file = '/data/gyf/newtest/ChangeFormer/checkpoint3/base_transformer_pos_s4_dd8_3/CD_base_transformer_pos_s4_dd8_LEVIR_b16_lr0.0001_adamw_train_test_201_linear_ce_multi_train_False_multi_infer_False_shuffle_AB_False_embed_dim_256/log.txt'
    output_csv_file = 'extracted_data_bit_withStyle.csv'
    
    if not os.path.exists(input_log_file):
        print(f"❌ 错误：未找到日志文件 '{input_log_file}'。")
    else:
        parse_log_file(input_log_file, output_csv_file)