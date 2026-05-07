#!/usr/bin/env python3
"""
all_series_compare.py — 3d/4d/5d 纯组态量子亏损三代对比 (修正版)
"""

import csv
import numpy as np
import re
import os
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# ============================================================
# 三代离子配置 (电离限已从文件末尾 Limit 行提取)
# ============================================================
ION_3D = [
    {"name": "Ti II", "Z": 22, "file": "TiII.csv", "z_eff": 2.0, "limit_fb": 109494, "n_shell": 3},
    {"name": "V II",  "Z": 23, "file": "VII.csv",  "z_eff": 2.0, "limit_fb": 118030, "n_shell": 3},
    {"name": "Cr II", "Z": 24, "file": "CrII.csv", "z_eff": 2.0, "limit_fb": 132971, "n_shell": 3},
    {"name": "Fe II", "Z": 26, "file": "FeII.csv", "z_eff": 2.0, "limit_fb": 130655, "n_shell": 3},
    {"name": "Cu II", "Z": 29, "file": "CuII.csv", "z_eff": 2.0, "limit_fb": 163669, "n_shell": 3},
]

ION_4D = [
    {"name": "Y III",  "Z": 39, "file": "YIII.csv",  "z_eff": 3.0, "limit_fb": 165540, "n_shell": 4},
    {"name": "Zr III", "Z": 40, "file": "ZrIII.csv", "z_eff": 3.0, "limit_fb": 186880, "n_shell": 4},
    {"name": "Nb III", "Z": 41, "file": "NbIII.csv", "z_eff": 3.0, "limit_fb": 202000, "n_shell": 4},
    {"name": "Mo III", "Z": 42, "file": "MoIII.csv", "z_eff": 3.0, "limit_fb": 218800, "n_shell": 4},
    {"name": "Ru III", "Z": 44, "file": "RuIII.csv", "z_eff": 3.0, "limit_fb": 229600, "n_shell": 4},
    {"name": "Rh III", "Z": 45, "file": "RhIII.csv", "z_eff": 3.0, "limit_fb": 250500, "n_shell": 4},
    {"name": "Pd III", "Z": 46, "file": "PdIII.csv", "z_eff": 3.0, "limit_fb": 265600, "n_shell": 4},
    {"name": "Ag III", "Z": 47, "file": "AgIII.csv", "z_eff": 3.0, "limit_fb": 280900, "n_shell": 4},
]

ION_5D = [
    {"name": "Lu II",  "Z": 71, "file": "LuII.csv",  "z_eff": 2.0, "limit_fb": 113970, "n_shell": 5},
    {"name": "Hf II",  "Z": 72, "file": "HfII.csv",  "z_eff": 2.0, "limit_fb": 117820, "n_shell": 5},
    {"name": "Ta II",  "Z": 73, "file": "TaII.csv",  "z_eff": 2.0, "limit_fb": 131000, "n_shell": 5},
    {"name": "W II",   "Z": 74, "file": "WII.csv",   "z_eff": 2.0, "limit_fb": 139000, "n_shell": 5},
    {"name": "Os III", "Z": 76, "file": "OsIII.csv", "z_eff": 3.0, "limit_fb": 202000, "n_shell": 5},
    {"name": "Ir III", "Z": 77, "file": "IrIII.csv", "z_eff": 3.0, "limit_fb": 226000, "n_shell": 5},
]

RYDBERG = 109737.3156
MAX_N = 10
PLOT_DIR = "plots_3d_4d_5d"
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# CSV 读取
# ============================================================
def read_nist_csv(filepath):
    records = []
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        config_idx, term_idx, energy_idx = 0, 1, 3
        if header:
            for i, col in enumerate(header):
                col_l = col.strip().lower()
                if 'conf' in col_l: config_idx = i
                elif 'term' in col_l: term_idx = i
                elif 'level' in col_l: energy_idx = i
        for row in reader:
            if len(row) <= energy_idx: continue
            config = row[config_idx].strip()
            term = row[term_idx].strip() if len(row) > term_idx else ""
            energy_raw = row[energy_idx]

            # 电离限行
            if 'limit' in config.lower() or 'limit' in term.lower():
                clean = re.sub(r'[^\d.]', '', energy_raw.replace(' ', ''))
                if clean:
                    records.append(('Limit', float(clean)))
                continue

            # 普通能级
            clean_e = re.sub(r'[^\d.]', '', energy_raw.replace(' ', '').replace('[', '').replace(']', ''))
            if clean_e:
                records.append(('Level', config, float(clean_e)))
    return records

# ============================================================
# 纯 d 组态判定 (修正版)
# ============================================================
def is_pure_d(config, series):
    """
    排除包含任何外层 s/p/d 轨道的混合组态。
    例如: 3d2(3F)4s 视为不纯。
    """
    s = config.lower()
    if f'{series}d' not in s:
        return False

    # 匹配任何数字开头的轨道符号 (如 4s, 5p, 6d)
    all_orbitals = re.findall(r'\b(\d+)([spdf])\b', s)
    for n_str, orb_type in all_orbitals:
        n = int(n_str)
        if n == series and orb_type == 'd':
            # 当前系列的 d 轨道：需要检查是否还有外层电子 (即整个组态中是否有另一个轨道)
            # 这条规则不够精确，改为：直接禁止出现更高主量子数或同主量子数的其他轨道。
            pass
        # 对于 series=3: 不允许出现 n>=4 的任何轨道
        if series == 3 and n >= 4:
            return False
        # 对于 series=4: 不允许出现 n>=5 的任何轨道 (但允许 n=4 的 d，禁止 5s 等)
        if series == 4 and n >= 5:
            return False
        # 对于 series=5: 不允许出现 n>=6 的任何轨道 (允许 5d，禁止 6s 等)
        if series == 5 and n >= 6:
            return False

    # 如果存在同主量子数的 s 或 p 轨道也要排除 (例如 4d.5s 组合中 5s 会被上面捕获，但 4d²4s¹ 这种 4s 也会被排除)
    # 上面的规则已经足够，因为 4s 的主量子数 4 >= 3+1? 实际上对于 3d 系列，4s 的 n=4 > 3，所以会被 series==3 时的 n>=4 过滤掉。
    # 这就正确了。
    return True

# ============================================================
# 分析函数
# ============================================================
def get_peak(deltas):
    if len(deltas) < 5: return None
    kde = gaussian_kde(deltas)
    x_grid = np.linspace(min(deltas)-0.05, max(deltas)+0.05, 200)
    return x_grid[np.argmax(kde(x_grid))]

def audit_series(ion_list, series):
    results = []
    for ion in ion_list:
        records = read_nist_csv(ion["file"])
        if not records: continue
        levels, limit = [], None
        for r in records:
            if r[0] == 'Limit': limit = r[1]
            else: levels.append((r[1], r[2]))
        if limit is None: limit = ion["limit_fb"]
        z_eff = ion["z_eff"]

        deltas = []
        current_config = ""
        for config, energy in levels:
            if not config: continue          # 跳过空配置行，不再继承
            current_config = config
            if not is_pure_d(current_config, series): continue

            # 提取主量子数
            n_match = re.search(r'(\d+)d', current_config)
            if not n_match: continue
            n = int(n_match.group(1))
            if n > MAX_N: continue

            term = limit - energy
            if term <= 0: continue
            n_eff = np.sqrt((RYDBERG * (z_eff**2)) / term)
            delta = n - n_eff
            if 0 <= delta <= n: deltas.append(delta)

        if len(deltas) < 5: continue
        peak = get_peak(deltas)
        if peak is None: continue
        print(f"  {ion['name']:10s} (Z={ion['Z']:2d})  {series}d  δ_peak = {peak:.4f}  (n={len(deltas)})")
        results.append((ion['name'], ion['Z'], peak, series))
    return results

# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 70)
    print("       3d / 4d / 5d 纯组态量子亏损三代对比 (修正版)")
    print("=" * 70)

    all_results = []
    print("\n--- 3d 系列 ---")
    all_results.extend(audit_series(ION_3D, 3))
    print("\n--- 4d 系列 ---")
    all_results.extend(audit_series(ION_4D, 4))
    print("\n--- 5d 系列 ---")
    all_results.extend(audit_series(ION_5D, 5))

    if not all_results:
        print("No results.")
        return

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {3: 'steelblue', 4: 'darkorange', 5: 'crimson'}
    for name, Z, peak, series in all_results:
        ax.scatter(Z, peak, color=colors[series], s=80, zorder=5)
        ax.annotate(name, (Z, peak), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='3d series', markerfacecolor='steelblue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='4d series', markerfacecolor='darkorange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='5d series', markerfacecolor='crimson', markersize=10),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_xlabel('Atomic Number Z')
    ax.set_ylabel('Quantum Defect δ peak')
    ax.set_title('Pure d‑orbital Quantum Defect: 3d vs 4d vs 5d')
    ax.grid(True, alpha=0.3)
    fname = os.path.join(PLOT_DIR, "3d_4d_5d_comparison.pdf")
    fig.savefig(fname)
    plt.close(fig)
    print(f"\n三代对比图已保存至 {fname}")

if __name__ == "__main__":
    main()