import os
import csv
import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt


def analyze_and_save_batch(image_folder, output_csv='brightness_results.csv', grid_size=(5,5), ref_image_path='reference.jpg'):
    """
    批量处理图像亮度分析并保存结果
    参数：
    image_folder: 图片文件夹路径
    output_csv: 结果保存文件名
    grid_size: 分析网格尺寸
    """
    ref_img = cv2.imread(ref_image_path)
    if ref_img is None:
        raise ValueError(f"无法读取标准图像文件: {ref_image_path}")
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    # ref_gray = cv2.resize(ref_gray, (640, 480))  # 根据实际尺寸调整

    # 创建热力图保存目录
    heatmap_dir = os.path.join(image_folder, 'uniformity_heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)

    # 准备结果存储
    results = []

    # 遍历图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    for filename in [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]:
        try:
            filepath = os.path.join(image_folder, filename)
            gray = cv2.imread(filepath)
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

            # 执行分析
            analysis = analyze_brightness_uniformity(filepath, grid_size)

            # 生成热力图
            heatmap_path = os.path.join(heatmap_dir, f"{os.path.splitext(filename)[0]}_heatmap.png")
            save_heatmap(analysis, heatmap_path)

            # 新增PSNR计算
            current_gray = cv2.resize(gray, (ref_gray.shape[1], ref_gray.shape[0]))
            mse = np.mean((ref_gray - current_gray) ** 2)
            psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse != 0 else float('inf')
            # 记录结果
            results.append({
                'filename': filename,
                'global_std': analysis['global_std'],
                'max_variation': analysis['max_variation'],
                'psnr': psnr,  # 新增PSNR字段
                'heatmap_path': heatmap_path
            })

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")

    # 保存CSV结果
    with open(os.path.join(image_folder, output_csv), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'global_std', 'max_variation', 'psnr', 'heatmap_path'])
        writer.writeheader()
        writer.writerows(results)


def save_heatmap(analysis, save_path):
    """保存热力图到文件"""
    plt.figure(figsize=(10, 10))
    plt.imshow(analysis['grid_stds'], cmap='hot', interpolation='nearest')
    plt.colorbar(label='Standard Deviation')
    plt.title('Local Brightness Variation')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


# 修改后的分析函数
def analyze_brightness_uniformity(image_path, grid_size=(5, 5)):
    """（保持原有分析逻辑，添加文件读取校验）"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 方法1：全局统计分析
    global_mean = np.mean(gray)
    global_std = np.std(gray)

    # 方法2：网格化局部分析
    rows, cols = gray.shape
    grid_rows, grid_cols = grid_size
    cell_height = rows // grid_rows
    cell_width = cols // grid_cols

    grid_means = np.zeros(grid_size)
    grid_stds = np.zeros(grid_size)

    for i in range(grid_rows):
        for j in range(grid_cols):
            y_start = i * cell_height
            y_end = (i + 1) * cell_height if i < grid_rows - 1 else rows
            x_start = j * cell_width
            x_end = (j + 1) * cell_width if j < grid_cols - 1 else cols

            cell = gray[y_start:y_end, x_start:x_end]
            grid_means[i, j] = np.mean(cell)
            grid_stds[i, j] = np.std(cell)

    # 计算最大亮度差异
    max_variation = np.max(grid_means) - np.min(grid_means)

    # 直方图分析
    hist = exposure.histogram(gray)

    return {
        'global_std': global_std,
        'grid_stds': grid_stds,
        'max_variation': max_variation,
        'histogram': hist,
        'source_path': image_path,
        'gray_image': gray  # 新增返回灰度图像
    }


# 使用示例
if __name__ == '__main__':
    analyze_and_save_batch(
        image_folder="D:/Data/vision/midspeedconsumption/cam",
        output_csv="brightness_analysis.csv",
        grid_size=(10, 10),
        ref_image_path="D:/Data/vision/midspeedconsumption/cam/5.png"
    )
