import cv2
import numpy as np
import os

from matplotlib import pyplot as plt

# --- 可配置参数 ---

# 1. 分析区域 (ROI) 的定义 [y_start, y_end, x_start, x_end]
# 我们选择一个避开顶部和底部螺丝以及中间孔洞的区域。
# 这里的坐标是基于你的示例图片（高约1200px）估算的。
# 你可以根据实际拍摄图像的尺寸和位置进行调整。
# 格式: [y_start_ratio, y_end_ratio, x_start_ratio, x_end_ratio]
ROI_RATIOS = (0.05, 0.2, 0.2, 0.8)


def identify_material(image_path, ROI_RATIOS):
    """
    根据图像ROI的颜色和纹理特性来区分金属和白色塑料。

    Args:
        image_path (str): 图像文件路径。
        ROI_RATIOS (tuple): 定义ROI的比例 (y_start, y_end, x_start, x_end)。

    Returns:
        str: 识别结果 ("White Plastic", "Metal", "Unknown") 或错误信息。
    """
    # 1. 读取彩色图像
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Could not read image {image_path}"

    # 2. 根据比例计算ROI的具体坐标
    h, w, d = image.shape
    y_start = int(h * ROI_RATIOS[0])
    y_end = int(h * ROI_RATIOS[1])
    x_start = int(w * ROI_RATIOS[2])
    x_end = int(w * ROI_RATIOS[3])

    roi = image[y_start:y_end, x_start:x_end]
    # 3. 在原始图像上绘制ROI区域
    image_with_roi = image.copy()
    cv2.rectangle(image_with_roi, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    # 4. 计算ROI区域的直方图
    # 分离三个颜色通道
    b, g, r = cv2.split(roi)

    # 计算每个通道的直方图
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    mode_b = np.argmax(hist_b)
    mode_g = np.argmax(hist_g)
    mode_r = np.argmax(hist_r)
    # 5. 创建直方图可视化
    plt.figure(figsize=(15, 10))

    # 显示原始图像和ROI区域
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image_with_roi, cv2.COLOR_BGR2RGB))
    plt.title('Original Image with ROI')
    plt.axis('off')

    # 显示ROI区域
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.title('ROI Region')
    plt.axis('off')

    # 显示RGB直方图
    plt.subplot(2, 2, 3)
    plt.plot(hist_r, color='red', label='Red')
    plt.plot(hist_g, color='green', label='Green')
    plt.plot(hist_b, color='blue', label='Blue')
    plt.title('Color Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)

    # 显示归一化直方图（更易比较）
    plt.subplot(2, 2, 4)
    plt.plot(hist_r / hist_r.sum(), color='red', label='Red')
    plt.plot(hist_g / hist_g.sum(), color='green', label='Green')
    plt.plot(hist_b / hist_b.sum(), color='blue', label='Blue')
    plt.title('Normalized Color Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return mode_b,mode_g,mode_r

# --- 主程序入口 ---
if __name__ == "__main__":
    # 请将 'v1_plastic.png' 和 'v2_metallic.png' 替换为你的实际文件名
    # 这里假设你的图片和脚本在同一个目录下
    img_folder = "E:/project_pycharm/MYMLOPs/comsuption_mid/dataset/buffer_new/images"
    # img_folder = "datasetv2/detection_20250702#12/areaB"
    for filename in os.listdir(img_folder):
        v1_image_path = os.path.join(img_folder, filename)
        mode_b,mode_g,mode_r = identify_material(v1_image_path, ROI_RATIOS)
        print(filename,mode_b,mode_g,mode_r)
