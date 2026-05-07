import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 数据集：使用 Slater 规则修正后的 Z*
# 格式: (Slater_Z, Delta_Peak, Status)
# Status: 1=有峰, 0=平坦
data = [
    (3.65, 1.0631, 1), # Ti II
    (4.30, 1.1694, 1), # V II
    (4.95, 1.1830, 1), # Cr II
    (6.30, 1.0794, 0), # Mn III
    (6.25, 1.1671, 1), # Fe II
    (6.60, 1.0016, 0), # Fe III
    (7.60, 0.9394, 0), # Co IV
    (7.90, 1.1345, 1), # Ni III
    (8.25, 0.6149, 0), # Ni IV
    (8.20, 1.3825, 1), # Cu II
]

z_stars = np.array([x[0] for x in data])
deltas = np.array([x[1] for x in data])
status = np.array([x[2] for x in data])

# 1. 寻找寻址相变边界
flat_zone = z_stars[status == 0]
peak_zone = z_stars[status == 1]
print(f">>> 寻址相变审计")
print(f"平坦区 Z* 范围: {flat_zone.min():.2f} ~ {flat_zone.max():.2f}")
print(f"有峰区 Z* 范围: {peak_zone.min():.2f} ~ {peak_zone.max():.2f}")

# 2. 核心审计：Z* 驱动的地址漂移线性度 (针对有峰区)
# 排除 Cu II (新页开启) 进行线性分析
mask = (status == 1) & (z_stars < 8.0)
slope, intercept, r_val, p_val, std_err = linregress(z_stars[mask], deltas[mask])

print(f"\n>>> 逻辑地址位移线性审计 (Z=22~28 有峰区)")
print(f"地址漂移斜率 (Slope): {slope:.4f} δ/Z*")
print(f"线性相关度 (R^2): {r_val**2:.4f}")

# 3. 验证 0.1 步进
# 如果斜率 * 平均 Z* 步增 ≈ 0.1，则 Stride-10 协议成立
avg_z_step = np.mean(np.diff(np.sort(z_stars[mask])))
logical_step = abs(slope * 1.0) # 每增加一个单位 Z* 的位移
print(f"单位寻址压差位移: {logical_step:.4f} (理论目标: 0.100)")

# 4. 解释 Ni III
ni3_z = 7.90
pred_ni3_delta = slope * ni3_z + intercept
print(f"\n>>> Ni III 反常点审计")
print(f"Ni III Slater Z*: {ni3_z}")
print(f"线性回归预测值: {pred_ni3_delta:.4f}")
print(f"实验观测 KDE 峰: 1.1345")
print(f"偏差: {abs(1.1345 - pred_ni3_delta):.4f}")