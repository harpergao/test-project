import matplotlib.pyplot as plt
import cv2
import os

# ==================== 1. 配置区域 (只需修改这里) ====================

# 定义每一列的标题 (显示在图片最上方)
COL_HEADERS = ["T1 image", "T2 image", "FC-EF", "Siam-diff", "Siam-conc", "ChangeFormer", "BIT", "DTCDSCN", "ELGCNet", "FACDNet", "Ground Truth"]

# 定义每一行的标题 (显示在图片最左侧，比如数据集名称)
# 如果某一行不想显示标题，填空字符串 "" 即可
ROW_HEADERS = ["SYSU-CD","", "WHU-CD","", "LEVIR-CD",""]

# 核心数据：请按【行】顺序填入图片的绝对路径
# 注意：每一行的路径数量必须和 COL_HEADERS 的数量一致！
# 建议在路径前加 r，防止转义字符报错
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

# ==================== 2. 绘图逻辑 (无需修改) ====================

def plot_manual_grid():
    nrows = len(FILE_GRID)
    ncols = len(FILE_GRID[0])
    
    # 检查行列标题数量是否匹配
    if len(COL_HEADERS) != ncols:
        print(f"警告：列标题数量({len(COL_HEADERS)}) 与 图片列数({ncols}) 不一致！")
    if len(ROW_HEADERS) != nrows:
        print(f"警告：行标题数量({len(ROW_HEADERS)}) 与 图片行数({nrows}) 不一致！")

    # 创建画布
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2.5, nrows * 2.5))
    plt.subplots_adjust(wspace=0.03, hspace=0.03) # 调整间隙

    for i in range(nrows):
        for j in range(ncols):
            # 获取当前的坐标轴对象
            if nrows == 1 and ncols == 1: ax = axes
            elif nrows == 1: ax = axes[j]
            elif ncols == 1: ax = axes[i]
            else: ax = axes[i, j]

            # 获取路径并读取
            img_path = FILE_GRID[i][j]
            
            if not os.path.exists(img_path):
                print(f"[错误] 找不到文件: {img_path}")
                # 显示一个白色空图
                img = 255 * sorted(os.urandom(256*256*3)) # 随便填，其实可以用np.ones
                import numpy as np
                img = np.ones((256,256,3), dtype=np.uint8) * 255
                # 在图中写字提示
                ax.text(0.5, 0.5, "File Not Found", ha='center', va='center', fontsize=8)
            else:
                img = cv2.imread(img_path)
                # BGR 转 RGB
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    print(f"[错误] 无法解码图片: {img_path}")
                    import numpy as np
                    img = np.zeros((256,256,3), dtype=np.uint8)

            ax.imshow(img)
            
            # 关闭刻度
            ax.set_xticks([])
            ax.set_yticks([])

            # --- 设置列标题 (仅第一行) ---
            if i == 0 and j < len(COL_HEADERS):
                ax.set_title(COL_HEADERS[j], fontsize=14, fontweight='bold', pad=8)
            
            # --- 设置行标题 (仅第一列) ---
            if j == 0 and i < len(ROW_HEADERS) and ROW_HEADERS[i]:
                ax.set_ylabel(ROW_HEADERS[i], fontsize=14, fontweight='bold', labelpad=10)

    save_path = "/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/Apple/manual_grid_result4.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"拼图完成！已保存为: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_manual_grid()