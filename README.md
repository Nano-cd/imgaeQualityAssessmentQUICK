
# Image Brightness and Uniformity Analysis Toolkit

This project provides a set of Python scripts to quantitatively analyze and visualize the brightness, texture, and uniformity of images in a batch. It is particularly useful for quality control tasks, such as comparing surface finishes in manufacturing (e.g., old vs. new sheet metal versions), or any scenario where consistent image quality is important.

The toolkit consists of two main parts:
1.  **Analysis Script (`analyze_and_save_batch`)**: This script processes a folder of images, calculates several key metrics for each, generates visual heatmaps of local variations, and saves all results to a CSV file.
2.  **Visualization Script (`boxplot_visualization`)**: This script provides an example of how to take the statistical results (e.g., from the CSV) and create a clear, annotated comparison plot (like a boxplot) to highlight differences between two groups of images.

## Features

-   **Batch Processing**: Automatically analyze all images (`.jpg`, `.png`, etc.) in a specified folder.
-   **Quantitative Metrics**: Calculates key indicators for each image:
    -   **Global Standard Deviation**: Measures overall image contrast/texture.
    -   **Maximum Variation**: Finds the difference in brightness between the brightest and darkest regions of an image.
    -   **Peak Signal-to-Noise Ratio (PSNR)**: Compares each image to a reference image to measure similarity and quality.
-   **Local Uniformity Analysis**: Divides each image into a grid to analyze local brightness variations.
-   **Heatmap Generation**: Creates a visual heatmap for each image, showing areas of high vs. low texture/variation.
-   **Data Export**: Saves all calculated metrics and file paths into a clean CSV file for further analysis in Excel, Python (Pandas), or other tools.
-   **Example Visualization**: Includes a script to generate a publication-quality boxplot to compare datasets, complete with annotations.

## Installation

1.  **Clone the repository or download the files.**

2.  **Install required Python libraries.** It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    If you don't have a `requirements.txt` file, you can install the packages manually:
    ```bash
    pip install opencv-python numpy scikit-image matplotlib
    ```

3.  **(Optional but Recommended) Font for Visualization**: The visualization script uses a specific Chinese font (`SimHei`) for its labels. If you don't have this font, `matplotlib` may show squares instead of characters.
    -   **Windows**: `SimHei` (黑体) is usually pre-installed.
    -   **macOS**: You can change `'SimHei'` in the script to `'PingFang SC'`.
    -   **Linux**: You may need to install a font like `wqy-zenhei` (`sudo apt-get install fonts-wqy-zenhei`) and then update `matplotlib`'s font cache.

## Usage

### Part 1: Analyzing Image Batches

The main analysis is performed by the first script containing the `analyze_and_save_batch` function.

#### 1. Prepare Your Folder Structure

Organize your files as follows:
```
my_project/
├── analysis_script.py
├── visualization_script.py
└── data/
    ├── reference.png              <-- Your 'perfect' or standard reference image
    ├── image_01.png
    ├── image_02.png
    └── ...
```

#### 2. Configure and Run the Analysis

Open the analysis script and modify the `if __name__ == '__main__':` block at the bottom:

```python
if __name__ == '__main__':
    analyze_and_save_batch(
        # 1. Path to the folder containing your images
        image_folder="data",

        # 2. Name of the output CSV file
        output_csv="brightness_analysis.csv",

        # 3. Grid size for local analysis (e.g., 10x10)
        grid_size=(10, 10),

        # 4. Path to your reference image for PSNR calculation
        ref_image_path="data/reference.png"
    )
```

Run the script from your terminal:
```bash
python analysis_script.py
```

#### 3. Review the Output

After the script finishes, you will find two new items inside your `image_folder` (`data/` in this example):

1.  **`brightness_analysis.csv`**: A CSV file with the results.
    | filename | global_std | max_variation | psnr | heatmap_path |
    | :--- | :--- | :--- | :--- | :--- |
    | image_01.png | 53.96 | 85.61 | 15.23 | data/uniformity_heatmaps/image_01_heatmap.png |
    | image_02.png | 3.13 | 13.23 | 28.71 | data/uniformity_heatmaps/image_02_heatmap.png |
    | ... | ... | ... | ... | ... |

2.  **`uniformity_heatmaps/`**: A new folder containing the heatmaps for each processed image. In a heatmap, "hotter" colors (like red and yellow) indicate regions with higher local standard deviation (more texture or brightness variation).


*Example of a heatmap where the center is more uniform (cooler color) and the edges have more variation (hotter colors).*

---

### Part 2: Visualizing the Results

The second script is a **template** for creating a comparative boxplot. It does **not** read the CSV file automatically. You need to input your summary statistics manually.

#### 1. Get Your Statistics

Open the generated `brightness_analysis.csv` file. For the two groups you want to compare (e.g., "V1" and "V2"), calculate the mean, standard deviation, min, and max of a relevant metric (e.g., `global_std`).

#### 2. Update and Run the Visualization Script

Open the visualization script and update the statistics dictionaries with your own data:

```python
# --- 2. 录入您图片中的统计数据 ---
# Update these values based on your CSV results

# Data for the first group (e.g., New Version)
v2_stats = {
    'mean': 3.137,     # <-- Your calculated mean
    'std_dev': 3.786,  # <-- Your calculated std dev
    'min': 0,          # <-- Your min value
    'max': 13.23       # <-- Your max value
}

# Data for the second group (e.g., Old Version)
v1_stats = {
    'mean': 53.966,
    'std_dev': 16.368,
    'min': 29.92,
    'max': 115.53
}
```
You can also change the labels and titles in the script to match your context.

Run the script:
```bash
python visualization_script.py
```

#### 3. View the Plot

The script will generate and display a boxplot that visually compares the two datasets. The example script is heavily annotated to show how to highlight key differences, such as a clear separation between the value ranges of the two groups.



## Understanding the Metrics

-   **Global Standard Deviation (`global_std`)**: A measure of the overall contrast or texture in the image. A higher value means more variation in brightness levels across the entire image (e.g., a rough or patterned surface). A low value suggests a more uniform, flat-colored image.
-   **Max Variation (`max_variation`)**: The difference between the average brightness of the brightest grid cell and the dimmest grid cell. This metric is excellent for detecting uneven lighting or large-scale blemishes.
-   **PSNR (Peak Signal-to-Noise Ratio)**: Compares the pixel values of an image against a "perfect" reference image. A higher PSNR value indicates that the image is more similar to the reference. An infinite PSNR means the images are identical. This is useful for detecting deviations from a standard.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
