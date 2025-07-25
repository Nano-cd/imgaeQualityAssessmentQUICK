import cv2
import numpy as np
import math
import os
import glob


def calculate_psnr(img1, img2):
    """
    计算两张8位图像的峰值信噪比（PSNR）。
    图像必须具有相同的尺寸。
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def create_comparison_visualization(image1, image2, psnr_value, display_dim=None):
    """
    创建并显示一个包含原图、差异图和热力图的统一画布。

    参数:
    image1 (np.array): 标准图像。
    image2 (np.array): 对比图像。
    psnr_value (float): 计算出的PSNR值。
    display_dim (tuple, optional): 用于显示的图片尺寸 (宽度, 高度)。
                                   如果为 None，则使用原图尺寸。
    """
    # 如果指定了显示尺寸，则以此为准；否则使用原图尺寸。
    if display_dim:
        disp_w, disp_h = display_dim
        print(f"为适应窗口，所有图片将被缩放至 {disp_w}x{disp_h} 进行显示。")
    else:
        orig_h, orig_w, _ = image1.shape
        disp_w, disp_h = orig_w, orig_h

    # --- 图像处理和差异计算 (在原图上进行) ---
    diff_abs = cv2.absdiff(image1, image2)
    diff_gray = cv2.cvtColor(diff_abs, cv2.COLOR_BGR2GRAY)
    diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

    # --- 准备用于显示的图片 (进行缩放) ---
    display_size = (disp_w, disp_h)
    img1_display = cv2.resize(image1, display_size, interpolation=cv2.INTER_AREA)
    img2_display = cv2.resize(image2, display_size, interpolation=cv2.INTER_AREA)
    diff_gray_display = cv2.resize(diff_gray, display_size, interpolation=cv2.INTER_AREA)
    heatmap_display = cv2.resize(diff_heatmap, display_size, interpolation=cv2.INTER_AREA)

    # --- 创建统一画布 (基于显示尺寸) ---
    padding = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, disp_w / 1000)  # 动态调整字体大小
    font_thickness = 1

    canvas = np.full((disp_h * 2 + padding * 3, disp_w * 2 + padding * 3, 3), 240, dtype=np.uint8)

    images_to_display = {
        "Standard Image": img1_display,
        "Comparison Image": img2_display,
        "Absolute Difference (Grayscale)": cv2.cvtColor(diff_gray_display, cv2.COLOR_GRAY2BGR),
        "Difference Heatmap": heatmap_display
    }

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (r, c), (title, img) in zip(positions, images_to_display.items()):
        y_offset = (r + 1) * padding + r * disp_h
        x_offset = (c + 1) * padding + c * disp_w
        canvas[y_offset:y_offset + disp_h, x_offset:x_offset + disp_w] = img
        cv2.putText(canvas, title, (x_offset, y_offset - 5), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # 在画布顶部显示PSNR值
    psnr_text = f"PSNR: {psnr_value:.2f} dB" if psnr_value != float('inf') else "PSNR: inf (Identical Images)"
    psnr_font_scale = max(0.8, disp_w / 800)
    psnr_font_thickness = 2
    text_size = cv2.getTextSize(psnr_text, font, psnr_font_scale, psnr_font_thickness)[0]
    text_x = (canvas.shape[1] - text_size[0]) // 2
    cv2.putText(canvas, psnr_text, (text_x, padding + 5), font, psnr_font_scale, (0, 0, 255), psnr_font_thickness,
                cv2.LINE_AA)

    cv2.imshow('Image Comparison', canvas)
    cv2.imwrite('comparison_result.jpg', canvas)
    print("可视化结果已保存为 'comparison_result.jpg'")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compare_single_pair(standard_path, comparison_path, resize_dim=None, visualize=True, display_dim=None):
    """
    比较两张指定的图片，计算PSNR，并可选择是否进行可视化及控制显示尺寸。
    """
    print(f"--- 正在比较两张图片 ---\n标准图: {standard_path}\n对比图: {comparison_path}")

    standard_img = cv2.imread(standard_path)
    comparison_img = cv2.imread(comparison_path)

    if standard_img is None or comparison_img is None:
        print("错误: 无法读取一张或多张图片，请检查路径。")
        return

    # 调整用于计算的图片尺寸 (如果需要)
    # PSNR计算必须在相同尺寸的图片上进行
    if resize_dim:
        standard_img_calc = cv2.resize(standard_img, resize_dim, interpolation=cv2.INTER_AREA)
    else:
        standard_img_calc = standard_img.copy()

    if standard_img_calc.shape != comparison_img.shape:
        comparison_img_calc = cv2.resize(comparison_img, (standard_img_calc.shape[1], standard_img_calc.shape[0]),
                                         interpolation=cv2.INTER_AREA)
    else:
        comparison_img_calc = comparison_img.copy()

    # 在尺寸统一的图片上计算PSNR
    psnr = calculate_psnr(standard_img_calc, comparison_img_calc)
    print(f"PSNR 值 (在 {standard_img_calc.shape[1]}x{standard_img_calc.shape[0]} 尺寸上计算): {psnr:.2f} dB")

    # 如果需要，则使用原始加载的图片进行可视化
    if visualize:
        print("正在生成可视化结果...")
        # 传递原始图片和显示尺寸参数
        create_comparison_visualization(standard_img, comparison_img, psnr, display_dim=display_dim)

    print("--- 比较完成 ---\n")


def batch_compare_folder(standard_path, folder_path, resize_dim=None):
    """
    将标准图片与指定文件夹中的所有图片进行比较，并输出PSNR列表。
    """
    print(f"--- 开始批量处理 ---\n标准图: {standard_path}\n文件夹: {folder_path}")

    standard_img = cv2.imread(standard_path)
    if standard_img is None:
        print(f"错误: 无法读取标准图片 '{standard_path}'。")
        return

    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在。")
        return

    if resize_dim:
        standard_img = cv2.resize(standard_img, resize_dim, interpolation=cv2.INTER_AREA)

    supported_formats = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    image_paths = []
    for fmt in supported_formats:
        image_paths.extend(glob.glob(os.path.join(folder_path, fmt)))

    if not image_paths:
        print("在指定文件夹中未找到支持的图片文件。")
        return

    results = {}

    for img_path in image_paths:
        if os.path.samefile(standard_path, img_path):
            continue

        comparison_img = cv2.imread(img_path)
        if comparison_img is None:
            print(f"警告: 跳过无法读取的文件 {os.path.basename(img_path)}")
            continue

        if standard_img.shape != comparison_img.shape:
            comparison_img = cv2.resize(comparison_img, (standard_img.shape[1], standard_img.shape[0]),
                                        interpolation=cv2.INTER_AREA)

        psnr = calculate_psnr(standard_img, comparison_img)
        results[os.path.basename(img_path)] = psnr

    print("\n--- 批量处理结果 (PSNR 在 " + (
        f"{resize_dim[0]}x{resize_dim[1]}" if resize_dim else "原图") + " 尺寸上计算) ---")
    if not results:
        print("没有可比较的图片。")
    else:
        sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
        for filename, psnr_val in sorted_results:
            print(f"{filename:<30} | PSNR: {psnr_val:.2f} dB")

    print("--- 批量处理完成 ---\n")
    return results


if __name__ == '__main__':
    # --- 配置区域 ---
    MODE = 'single'  # 'single' 或 'batch'
    STANDARD_IMAGE_PATH = '1_interface_20250714111109156-DP8.jpg.jpg'

    # --- 'single' 模式配置 ---
    COMPARISON_IMAGE_PATH = '1_interface_20250714111125348-DP9.jpg.jpg'

    # --- 'batch' 模式配置 ---
    FOLDER_PATH = 'D:/Data/vision/midspeedconsumption/TestDataset/#7wendingxing/b'  # <-- 修改为你的图片文件夹路径

    # --- 通用配置 ---
    # 1. 计算时统一调整尺寸 (可选, 用于PSNR计算)。设为 None 使用原图尺寸。
    CALCULATION_RESIZE_DIM = None  # 例如: (1280, 720)

    # 2. 可视化时每张图片的显示尺寸 (可选, 用于控制窗口大小)。设为 None 使用原图尺寸。
    VISUAL_DISPLAY_DIM = (600, 450)  # <-- 在此设置期望的显示尺寸！

    # --- 执行 ---
    if MODE == 'single':
        compare_single_pair(
            STANDARD_IMAGE_PATH,
            COMPARISON_IMAGE_PATH,
            resize_dim=CALCULATION_RESIZE_DIM,
            visualize=True,
            display_dim=VISUAL_DISPLAY_DIM  # 传递显示尺寸参数
        )
    elif MODE == 'batch':
        if not os.path.exists(FOLDER_PATH) or not os.path.isdir(FOLDER_PATH):
            print(f"错误：批量处理模式需要一个存在的文件夹。请创建 '{FOLDER_PATH}' 并放入图片，或修改 FOLDER_PATH 变量。")
        else:
            batch_compare_folder(
                STANDARD_IMAGE_PATH,
                FOLDER_PATH,
                resize_dim=CALCULATION_RESIZE_DIM
            )
    else:
        print(f"错误: 未知的模式 '{MODE}'。请选择 'single' 或 'batch'。")
