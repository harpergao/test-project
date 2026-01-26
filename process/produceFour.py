import cv2
import numpy as np
import os

# ==================== 1. 配置区域 ====================
# 输入大图路径 (请修改为你实际的图片路径)
LARGE_IMAGE_PATH = "/mnt/data1/gyf/code/newtest/ChangeFormer/vis6/WHU_FreqFormer/WHU_FreqFormer/eval_1.jpg" 

# 输出文件夹路径
OUTPUT_DIR = "/mnt/data1/gyf/code/newtest/ChangeFormer/ooutputdir/W_FF2"

# 小图尺寸
PATCH_SIZE = 256

# 定义行映射关系字典 {预测行索引 : 标签行索引}
# 索引从0开始计数。
# "第5行对应的标签位于第7行" -> {4: 6}
# "第六行对应的标签位于第8行" -> {5: 7}
ROW_MAPPING = {
    4: 6,
    5: 7

}
# ====================================================


def generate_error_visualization(pred_patch_gray, label_patch_gray):
    """
    核心功能：输入两个灰度小图块，输出彩色的误差可视化图 (RGB)。
    """
    h, w = pred_patch_gray.shape
    
    # --- 关键步骤：二值化 ---
    # 将灰度图（可能像素值是0和255，也可能是0-255之间）强制转换为标准的 0 和 1
    # 阈值设为 127，大于127变为1，小于等于127变为0
    _, pred_bin = cv2.threshold(pred_patch_gray, 127, 1, cv2.THRESH_BINARY)
    _, label_bin = cv2.threshold(label_patch_gray, 127, 1, cv2.THRESH_BINARY)
    
    # 转换为布尔类型方便逻辑运算
    pred = pred_bin.astype(bool)
    gt = label_bin.astype(bool)

    # 初始化输出彩色图像 (RGB顺序，方便后续处理，最后保存时转BGR)
    vis_image = np.zeros((h, w, 3), dtype=np.uint8)

    # --- 定义颜色 (RGB) ---
    COLOR_WHITE = [255, 255, 255] # TP (预测对，有变化)
    COLOR_GREEN = [86, 135, 146]     # FP (误报，预测有但实际无)86, 135, 146
    COLOR_RED   = [121, 60, 56]     # FN (漏报，预测无但实际有) 
    # TN (背景) 保持黑色 [0,0,0]

    # --- 计算逻辑掩膜并上色 ---
    vis_image[pred & gt] = COLOR_WHITE    # TP
    vis_image[pred & (~gt)] = COLOR_GREEN # FP
    vis_image[(~pred) & gt] = COLOR_RED   # FN

    return vis_image


def process_and_visualize(large_img_path, output_dir, row_mapping, patch_size):
    # 1. 准备工作
    if not os.path.exists(large_img_path):
        print(f"错误：找不到文件 {large_img_path}")
        return
    os.makedirs(output_dir, exist_ok=True)
    print(f"开始处理，结果将保存至: {output_dir}/")

    # 2. 以【灰度模式】读取大图
    # 这一点很重要，因为我们只需要亮度信息来判断是前景还是背景
    large_img_gray = cv2.imread(large_img_path, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = large_img_gray.shape
    num_cols = img_w // patch_size # 计算总列数 (通常是8)

    total_processed = 0

    # 3. 开始循环处理映射对
    for pred_row_idx, label_row_idx in row_mapping.items():
        print(f"--> 正在处理配对行: 预测行[索引{pred_row_idx}] vs 标签行[索引{label_row_idx}]")
        
        # 遍历每一列
        for col_idx in range(num_cols):
            # 计算当前列的X坐标范围 (预测和标签的X范围是一样的)
            x_start = col_idx * patch_size
            x_end = (col_idx + 1) * patch_size

            # --- A. 切割预测图 Patch ---
            y_start_p = pred_row_idx * patch_size
            y_end_p = (pred_row_idx + 1) * patch_size
            # numpy切片: img[y范围, x范围]
            pred_patch = large_img_gray[y_start_p:y_end_p, x_start:x_end]

            # --- B. 切割标签图 Patch ---
            y_start_l = label_row_idx * patch_size
            y_end_l = (label_row_idx + 1) * patch_size
            label_patch = large_img_gray[y_start_l:y_end_l, x_start:x_end]

            # --- C. 生成可视化结果 ---
            # 传入的patch已经是灰度图了
            vis_result_rgb = generate_error_visualization(pred_patch, label_patch)

            # --- D. 保存结果 ---
            # 文件名格式：Vis_P[预测行索引]_L[标签行索引]_Col[列索引].png
            filename = f"Vis_P{pred_row_idx}_L{label_row_idx}_Col{col_idx}.png"
            save_path = os.path.join(output_dir, filename)
            
            # OpenCV 保存图片需要 BGR 格式，所以这里要做一次转换
            vis_result_bgr = cv2.cvtColor(vis_result_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, vis_result_bgr)
            
            total_processed += 1
            # print(f"   已生成: {filename}") # 如果觉得刷屏太多可以注释掉

    print(f"\n处理完成！共生成 {total_processed} 张可视化结果图。")


# ================= 主程序入口 =================
if __name__ == "__main__":
    # 确保这里的大图路径是你自己的文件名
    # 假设你当前目录下有一个叫 test_grid.png 的大图
    # 如果没有，请先准备好图片并修改 LARGE_IMAGE_PATH
    
    # 简单创建一个模拟大图用于测试 (如果你已有真实图片，注释掉下面这三行)
    # dummy_img = np.zeros((2048, 2048), dtype=np.uint8)
    # cv2.putText(dummy_img, "Please replace with real image", (100,1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (255), 5)
    # cv2.imwrite(LARGE_IMAGE_PATH, dummy_img)
    
    process_and_visualize(LARGE_IMAGE_PATH, OUTPUT_DIR, ROW_MAPPING, PATCH_SIZE)