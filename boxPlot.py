import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# --- 1. 设置中文字体 (确保你的系统中有支持中文的字体，如黑体、宋体等) ---
# 如果你使用的是Windows系统, 'SimHei' 通常是可用的
# 如果是macOS, 'PingFang SC' 或 'Heiti TC' 都可以
# 如果是Linux, 你可能需要安装 wqy-zenhei 等字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题
except Exception as e:
    print("中文字体设置失败，请确保你的系统中安装了'SimHei'字体或更换为其他可用字体。")
    print(e)


# --- 2. 录入您图片中的统计数据 ---
# 数据源 V2
v2_stats = {
    'mean': 3.137156,
    'std_dev': 3.78697,
    'min': 0,
    'max': 13.23
}

# 数据源 V1
v1_stats = {
    'mean': 53.96636,
    'std_dev': 16.36896,
    'min': 29.92,
    'max': 115.53
}

# --- 3. 生成模拟数据 ---
# 我们生成1000个数据点来模拟原始分布
# np.random.normal(loc=均值, scale=标准差, size=数量)
np.random.seed(42) # 设置随机种子以保证每次运行结果一致
data_v2 = np.random.normal(loc=v2_stats['mean'], scale=v2_stats['std_dev'], size=1000)
# 由于原始数据有明确的min/max，我们对模拟数据进行裁剪，使其更符合实际情况
data_v2 = np.clip(data_v2, v2_stats['min'], v2_stats['max'])

data_v1 = np.random.normal(loc=v1_stats['mean'], scale=v1_stats['std_dev'], size=1000)
data_v1 = np.clip(data_v1, v1_stats['min'], v1_stats['max'])


# --- 4. 绘制箱型图 ---
fig, ax = plt.subplots(figsize=(10, 7))

# 要绘制的数据列表
data_to_plot = [data_v2, data_v1]

# 绘制箱型图
bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.6)

# --- 5. 美化和添加注释 (这是让图表清晰的关键) ---

# 设置箱体颜色
colors = ['#4169E1', '#FF6347'] # 皇家蓝 和 番茄红
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 设置中位数线的颜色和线宽
for median in bp['medians']:
    median.set(color='black', linewidth=2)

# 设置坐标轴和标题
ax.set_xticklabels(['V2 (新版本钣金)', 'V1 (旧版本钣金)'], fontsize=14)
ax.set_ylabel('平均局部标准差 (纹理复杂度)', fontsize=14)
ax.set_title('V1 与 V2 纹理复杂度对比箱型图', fontsize=18, pad=20)
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

# 添加关键注释来强调“无重叠”
# V2 的最大值
v2_max_val = v2_stats['max']
# V1 的最小值
v1_min_val = v1_stats['min']

# 绘制水平线来标记 V2 的上限和 V1 的下限
ax.axhline(y=v2_max_val, color='blue', linestyle='--', linewidth=1.5)
ax.text(0.55, v2_max_val + 1, f'V2 最大值: {v2_max_val}', color='blue', va='bottom', ha='center', fontsize=12)

ax.axhline(y=v1_min_val, color='red', linestyle='--', linewidth=1.5)
ax.text(1.55, v1_min_val - 1, f'V1 最小值: {v1_min_val}', color='red', va='top', ha='center', fontsize=12)

# 添加一个箭头和文本来突出显示“分离区域”
arrow_y_start = (v2_max_val + v1_min_val) / 2
ax.annotate('无数据重叠区域',
            xy=(1, arrow_y_start + 3),
            xytext=(1, arrow_y_start - 3),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2),
            ha='center', va='center', fontsize=14, color='green',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))


# 显示图表
plt.tight_layout()
plt.show()
