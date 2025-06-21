# imgaeQualityAssessmentQUICK
用于简单对比不同相机在拍摄相同主体下的一些基本亮度灰度信息噪声对比

# Brightness Uniformity Analyzer

This Python tool performs **batch brightness uniformity analysis** on images in a specified folder, calculates **local and global brightness variations**, generates **heatmaps**, and compares each image to a **reference image** using PSNR (Peak Signal-to-Noise Ratio). Analysis results are saved as a CSV file.

## 🔍 Features

- Batch process all images in a folder
- Grid-based local brightness analysis
- Global brightness standard deviation computation
- PSNR comparison with a reference image
- Heatmap generation for visual inspection
- CSV report output with key metrics

## 📁 Folder Structure
project/
│
├── brightness_analysis.py # Main script
├── reference.jpg # Reference image for PSNR
├── your_image_folder/
│ ├── image1.png
│ ├── image2.jpg
│ └── ...
│
└── your_image_folder/uniformity_heatmaps/
├── image1_heatmap.png
└── ...
## ⚙️ Requirements

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- scikit-image
- Matplotlib

You can install the dependencies with:

```bash
pip install opencv-python numpy scikit-image matplotlib
```


🚀 Usage
Modify the image_folder and ref_image_path paths in the if __name__ == '__main__': block at the bottom of brightness_analysis.py.

Run the script:

bash
复制
编辑
python brightness_analysis.py
A CSV file (brightness_analysis.csv) will be generated in your image folder, containing:

filename	global_std	max_variation	psnr	heatmap_path

Each image will also have a corresponding heatmap saved under uniformity_heatmaps/.

📊 Output Example
CSV Sample Row:

filename	global_std	max_variation	psnr	heatmap_path
5.png	12.45	34.87	41.23	uniformity_heatmaps/5_heatmap.png

Heatmap Example:

<p align="center"> <img src="your_image_folder/uniformity_heatmaps/sample_heatmap.png" width="400" /> </p>
📌 Notes
The grid size can be customized via grid_size=(rows, cols) for finer or coarser local analysis.

The reference image must have the same dimensions or will be automatically resized.

PSNR is used to evaluate similarity between the analyzed image and the reference image.

🧠 Function Overview
analyze_and_save_batch(...): Orchestrates the batch analysis.

analyze_brightness_uniformity(...): Computes brightness stats and grid-based metrics.

save_heatmap(...): Saves the visual heatmap of brightness variation.

🧪 Example Customization
python
复制
编辑
analyze_and_save_batch(
    image_folder="path/to/images",
    output_csv="my_results.csv",
    grid_size=(8, 8),
    ref_image_path="path/to/reference.jpg"
)
📝 License
MIT License

Developed with ❤️ for image quality inspection tasks.
