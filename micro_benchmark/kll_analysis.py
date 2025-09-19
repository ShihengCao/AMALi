import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# === 读入你的 CSV ===
df = pd.read_csv("kll_avg.csv")

# 保存每个 block size 的斜率
slopes = []
block_sizes = []

# 图1：Grid Size vs avg（按 block size 分组）
plt.figure(figsize=(12, 8))
unique_blocks = sorted(df["Block Size"].unique())

for i, bsize in enumerate(unique_blocks):
    group = df[df["Block Size"] == bsize]
    X = group["Grid Size"].values.reshape(-1, 1)
    y = group["avg"].values

    # 线性拟合
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)

    slopes.append(slope)
    block_sizes.append(bsize)

    # 绘制散点 + 拟合直线
    plt.scatter(X, y, label=f"Block {bsize}", alpha=0.6)
    plt.plot(X, y_pred, label=f"Fit {bsize}, slope={slope:.2f}, R²={r2:.3f}")

plt.xlabel("Grid Size")
plt.ylabel("Kernel Launch Latency (cycles)")
plt.title("Grid Size vs Latency (Linear Fit per Block Size)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grid_vs_latency.png", dpi=300)

# === 图2：Block Size vs 斜率（二次拟合） ===
block_sizes = [float(b)/32 for b in block_sizes]
X = np.array(block_sizes).reshape(-1, 1)
y = np.array(slopes)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

quad_model = LinearRegression()
quad_model.fit(X_poly, y)
y_pred = quad_model.predict(X_poly)

r2 = r2_score(y, y_pred)

a, b, c = quad_model.coef_[2], quad_model.coef_[1], quad_model.intercept_
print(f"二次拟合公式: slope = {a:.4e} * block_size^2 + {b:.4e} * block_size + {c:.4e}")
print(f"R² = {r2:.4f}")

# 绘制图形
plt.figure(figsize=(10, 6))

# 绘制原始数据点
plt.scatter(block_sizes, y, color='red', s=50, alpha=0.7, label='data', zorder=3)

# 为了绘制平滑的拟合曲线，生成更多的点
x_smooth = np.linspace(min(block_sizes), max(block_sizes), 100).reshape(-1, 1)
x_smooth_poly = poly.transform(x_smooth)
y_smooth = quad_model.predict(x_smooth_poly)

# 绘制拟合曲线
plt.plot(x_smooth, y_smooth, color='blue', linewidth=2, 
         label=f'quad fit (R² = {r2:.4f})', zorder=2)

# 设置图形属性
plt.xlabel('Block Size (warp count)', fontsize=12)
plt.ylabel('Slope', fontsize=12)
plt.title('Block Size vs quad fit', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, zorder=1)
plt.legend()

# 添加拟合公式到图中
formula_text = f'y = {a:.3e}x² + {b:.3e}x + {c:.3e}'
plt.text(0.05, 0.95, formula_text, transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
         fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.show()

# 可选：保存图片
plt.savefig('block_vs_slope.png', dpi=300, bbox_inches='tight')