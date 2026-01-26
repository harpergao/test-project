import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# ==================== 1. 配置区域 (只需修改这里) ====================

# [设置] 每一列的顶部标题
COL_HEADERS = ["T1 image", "T2 image", "FC-EF", "Siam-diff", "Siam-conc", "ChangeFormer", "BIT", "DTCDSCN", "ELGCNet", "FACDNet", "Label"]

# [设置] 左侧数据集名称 (只写名字即可)
DATASET_NAMES = ["SYSU-CD", "WHU-CD", "LEVIR-CD"]

# [设置] 每个数据集占几行？ (你现在的图片是每两行为一个数据集，所以填 2)
ROWS_PER_DATASET = 2 

# [样式] 左侧标签的字体大小
LABEL_FONT_SIZE = 20  
# [样式] 左侧标签的加粗程度 ('normal', 'bold', 'heavy')
LABEL_WEIGHT = 'bold' 
# [样式] 左侧标签距离图片的水平距离 (越小越靠右，负数表示在图片左边)
LABEL_X_OFFSET = 0.01 

# [数据] 图片路径列表 (请确保路径正确，逗号不要漏！)
# 这里的行数必须等于 len(DATASET_NAMES) * ROWS_PER_DATASET
# 例如：3个数据集 * 每个2行 = 总共6行路径
FILE_GRID = [
            # --- 第 5 行 (SYSU 样本) ---
    [
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/SYSU/Pred_R0_Col0.png",# Image A
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/SYSU/Label_R2_Col0.png",# Image B
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_Unet/Vis_P4_L6_Col0.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_SiamUnet_diff/Vis_P4_L6_Col0.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_SiamUnet_conc/Vis_P4_L6_Col0.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_change/Vis_P4_L6_Col0.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_BIT/Vis_P4_L6_Col0.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_DTCDSCN/Vis_P4_L6_Col0.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_ELGCNet/Vis_P4_L6_Col0.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_FF/Vis_P4_L6_Col0.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/SYSU/Label_R6_Col0.png",
    ],
            # --- 第 6 行 (SYSU 样本) ---
    [
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/SYSU/Pred_R0_Col2.png",# Image A
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/SYSU/Label_R2_Col2.png",# Image B
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_Unet/Vis_P4_L6_Col2.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_SiamUnet_diff/Vis_P4_L6_Col2.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_SiamUnet_conc/Vis_P4_L6_Col2.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_change/Vis_P4_L6_Col2.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_BIT/Vis_P4_L6_Col2.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_DTCDSCN/Vis_P4_L6_Col2.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_ELGCNet/Vis_P4_L6_Col2.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_FF/Vis_P4_L6_Col2.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/SYSU/Label_R6_Col2.png",
    ],
                # --- 第 6 行 (SYSU 样本) ---
    # [
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/SYSU/Pred_R0_Col6.png",# Image A
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/SYSU/Label_R2_Col6.png",# Image B
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_Unet/Vis_P4_L6_Col6.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_SiamUnet_diff/Vis_P4_L6_Col6.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_SiamUnet_conc/Vis_P4_L6_Col6.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_change/Vis_P4_L6_Col6.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_BIT/Vis_P4_L6_Col6.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_DTCDSCN/Vis_P4_L6_Col6.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_ELGCNet/Vis_P4_L6_Col6.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/S_FF/Vis_P4_L6_Col6.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/SYSU/Label_R6_Col6.png",
    # ],
        # --- 第 3 行 (WHU 样本) ---
    [
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_Crops/Pred_R0_Col4.png",# Image A
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_Crops/Label_R2_Col4.png",# Image B
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_Unet/Vis_P4_L6_Col4.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_SiamUnet_diff/Vis_P4_L6_Col4.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_SiamUnet_conc/Vis_P4_L6_Col4.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_change/Vis_P4_L6_Col4.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_BIT/Vis_P4_L6_Col4.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_DTCDSCN/Vis_P4_L6_Col4.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_ELGCNet/Vis_P4_L6_Col4.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_FF/Vis_P4_L6_Col4.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_Crops/Label_R6_Col4.png",
    ],
        # --- 第 4 行 (WHU 样本) ---
    [
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_Crops/Pred_R0_Col1.png",# Image A
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_Crops/Label_R2_Col1.png",# Image B
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_Unet/Vis_P4_L6_Col1.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_SiamUnet_diff/Vis_P4_L6_Col1.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_SiamUnet_conc/Vis_P4_L6_Col1.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_change/Vis_P4_L6_Col1.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_BIT/Vis_P4_L6_Col1.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_DTCDSCN/Vis_P4_L6_Col1.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_ELGCNet/Vis_P4_L6_Col1.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_FF/Vis_P4_L6_Col1.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_Crops/Label_R6_Col1.png",
    ],
        # --- 第 4 行 (WHU 样本) ---
    # [
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_Crops/Pred_R0_Col3.png",# Image A
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_Crops/Label_R2_Col3.png",# Image B
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_Unet/Vis_P4_L6_Col3.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_SiamUnet_diff/Vis_P4_L6_Col3.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_SiamUnet_conc/Vis_P4_L6_Col3.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_change/Vis_P4_L6_Col3.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_BIT/Vis_P4_L6_Col3.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_DTCDSCN/Vis_P4_L6_Col3.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_ELGCNet/Vis_P4_L6_Col3.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_FF/Vis_P4_L6_Col3.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_Crops/Label_R6_Col3.png",
    # ],

    # --- 第 1 行 (LEVIR-CD 样本) ---
    [
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/LEVIR/Pred_R0_Col0.png",            # Image A
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/LEVIR/Label_R2_Col0.png",            # Image B
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_Unet1/Vis_P4_L6_Col0.png",       # FC-EF (红绿图)      
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_SiamUnet_diff1/Vis_P4_L6_Col0.png",      # Siam-diff (红绿图)
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_SiamUnet_conc1/Vis_P4_L6_Col0.png" ,        # Siam-conc (红绿图)
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_ChangeFormer1/Vis_P4_L6_Col0.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_BIT/Vis_P4_L6_Col0.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_DTCDSCN1/Vis_P4_L6_Col0.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_ELGCNet1/Vis_P4_L6_Col0.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_FF1/Vis_P4_L6_Col0.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/LEVIR/Label_R6_Col0.png",        # Label
        
    ],
    
    # --- 第 2 行  ---
    [
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/LEVIR/Pred_R0_Col5.png",            # Image A
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/LEVIR/Label_R2_Col5.png",            # Image B
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_Unet1/Vis_P4_L6_Col5.png",       
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_SiamUnet_diff1/Vis_P4_L6_Col5.png",      
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_SiamUnet_conc1/Vis_P4_L6_Col5.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_ChangeFormer1/Vis_P4_L6_Col5.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_BIT/Vis_P4_L6_Col5.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_DTCDSCN1/Vis_P4_L6_Col5.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_ELGCNet1/Vis_P4_L6_Col5.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_FF1/Vis_P4_L6_Col5.png",
        r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/LEVIR/Label_R6_Col5.png",        # Label
        
    ],
        # --- 第 2 行  ---
    # [
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/LEVIR/Pred_R1_Col0.png",            # Image A
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/LEVIR/Label_R3_Col0.png",            # Image B
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_Unet1/Vis_P5_L7_Col0.png",       
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_SiamUnet_diff1/Vis_P5_L7_Col0.png",      
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_SiamUnet_conc1/Vis_P5_L7_Col0.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_ChangeFormer1/Vis_P5_L7_Col0.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_BIT/Vis_P5_L7_Col0.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_DTCDSCN1/Vis_P5_L7_Col0.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_ELGCNet1/Vis_P5_L7_Col0.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/L_FF1/Vis_P5_L7_Col0.png",
    #     r"/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/LEVIR/Label_R7_Col0.png",        # Label
        
    # ],


]

# ==================== 2. 绘图逻辑 (自动居中计算) ====================

def plot_centered_grid():
    nrows = len(FILE_GRID)
    ncols = len(FILE_GRID[0])
    
    # 检查行数是否匹配配置
    expected_rows = len(DATASET_NAMES) * ROWS_PER_DATASET
    if nrows != expected_rows:
        print(f"[警告] 配置了 {len(DATASET_NAMES)} 个数据集，每个 {ROWS_PER_DATASET} 行，应该有 {expected_rows} 行路径，但实际只有 {nrows} 行！")

    # 创建画布
    # figsize 可以根据需要微调，(列数*2, 行数*2) 是个不错的起点
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2.0, nrows * 2.0))
    
    # 调整子图间隙 (wspace是左右间距, hspace是上下间距)
    # 如果想让两行属于同一个数据集的图片靠得更近，可以把 hspace 设小一点
    plt.subplots_adjust(wspace=0.03, hspace=0.03) 

    # --- 1. 绘制图片 ---
    for i in range(nrows):
        for j in range(ncols):
            # 获取 ax
            if nrows == 1 and ncols == 1: ax = axes
            elif nrows == 1: ax = axes[j]
            elif ncols == 1: ax = axes[i]
            else: ax = axes[i, j]

            # 读取图片
            img_path = FILE_GRID[i][j]
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = np.ones((256,256,3), dtype=np.uint8) * 255 # 读不到给白图
            else:
                img = np.ones((256,256,3), dtype=np.uint8) * 255 # 路径不存在给白图
                
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            # 关闭边框 (可选)
            # ax.axis('off')

            # 设置顶部标题 (仅第一行)
            if i == 0:
                ax.set_title(COL_HEADERS[j], fontsize=12, fontweight='bold', pad=8)

    # --- 2. 核心：计算跨行居中标签 ---
    # 我们不使用 ax.set_ylabel，而是使用 fig.text
    
    # 获取最左侧一列的所有 ax 对象 (用于计算位置)
    # 假设 axes 是二维数组
    left_axes = axes[:, 0] if nrows > 1 else [axes[0]]

    for idx, name in enumerate(DATASET_NAMES):
        # 计算该数据集对应的起始行和结束行索引
        start_row_idx = idx * ROWS_PER_DATASET
        end_row_idx = start_row_idx + ROWS_PER_DATASET - 1
        
        # 保护机制：防止索引越界
        if end_row_idx >= nrows: break
        
        # 获取这组数据的【最上面一张图】和【最下面一张图】的坐标轴对象
        ax_top = left_axes[start_row_idx]
        ax_bottom = left_axes[end_row_idx]
        
        # 获取它们在整个画布中的位置 (Bounding Box)
        # pos 包含 x0(左), y0(下), x1(右), y1(上)
        pos_top = ax_top.get_position()
        pos_bottom = ax_bottom.get_position()
        
        # 计算垂直中心点：(第一行的上边缘 + 最后一样的下边缘) / 2
        y_center = (pos_top.y1 + pos_bottom.y0) / 2
        
        # 计算水平位置：以第一列的左边缘为基准，向左偏移
        x_pos = pos_top.x0 - LABEL_X_OFFSET
        
        # 在画布上添加文字
        fig.text(x_pos, y_center, name, 
                 va='center',   # 垂直居中
                 ha='center',   # 水平居中
                 rotation=90,   # 旋转90度
                 fontsize=LABEL_FONT_SIZE, 
                 fontweight=LABEL_WEIGHT)

    # 保存
    save_path = "/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/Apple/manual_grid_result5.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"完成！已保存为: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_centered_grid()