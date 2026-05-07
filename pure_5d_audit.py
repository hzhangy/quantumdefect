#!/usr/bin/env python3
"""
pure_5d_audit.py — 5d 纯组态量子亏损审计
仅保留价层为纯 5dⁿ (如 5d², 5d³) 的能级，
排除混合了 6s, 6p, 5f 等的组态，用于独立审计 5d 系列的寻址行为。
"""

import csv
import numpy as np
import re
import os
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# ============================================================
# 5d 离子配置 (可下载以下 CSV 后使用)
# ============================================================
ION_LIST = [
    # 下面这些离子具有较多纯 5dⁿ 组态
    {"name": "Lu II",  "Z": 71, "file": "LuII.csv",  "z_eff": 2.0, "limit_fb": 113970},
    {"name": "Hf II",  "Z": 72, "file": "HfII.csv",  "z_eff": 2.0, "limit_fb": 117820},
    {"name": "Ta II",  "Z": 73, "file": "TaII.csv",  "z_eff": 2.0, "limit_fb": 131000},
    {"name": "W II",   "Z": 74, "file": "WII.csv",   "z_eff": 2.0, "limit_fb": 139000},  # 数据可能不全
    {"name": "Os II",  "Z": 76, "file": "OsII.csv",  "z_eff": 2.0, "limit_fb": 137000},  # 数据质量较差
    {"name": "Ir II",  "Z": 77, "file": "IrII.csv",  "z_eff": 2.0, "limit_fb": 137100},
    {"name": "Pt II",  "Z": 78, "file": "PtII.csv",  "z_eff": 2.0, "limit_fb": 85745},   # 基态是4f⁵d，不适合纯5d审计，可跳过
    {"name": "Au II",  "Z": 79, "file": "AuII.csv",  "z_eff": 2.0, "limit_fb": 162950},
]

RYDBERG = 109737.3156
MAX_N = 10
PLOT_DIR = "plots_pure_5d"
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# CSV 读取 (兼容 NIST 方括号格式)
# ============================================================
def read_nist_csv(filepath):
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            config_idx, term_idx, energy_idx = 0, 1, 3
            if header:
                for i, col in enumerate(header):
                    col_l = col.strip().lower()
                    if 'conf' in col_l:
                        config_idx = i
                    elif 'term' in col_l:
                        term_idx = i
                    elif 'level' in col_l:
                        energy_idx = i
            for row in reader:
                if len(row) <= max(config_idx, energy_idx):
                    continue
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
    except FileNotFoundError:
        return None
    return records

# ============================================================
# 纯 5d 组态判定：必须含 "5d"，但之后不能紧跟 ".6s" 或 ".6p" 或 ".5f"
# ============================================================
def is_pure_5d(config):
    s = config.lower()
    if 'd' not in s:
        return False
    # 排除价层混合：5d.6s, 5d.6p, 4f 等
    if re.search(r'5d\.6[sp]', s):
        return False
    # 排除明显含 4f 或 5f 的组态（如 Pt II 基态）
    if re.search(r'4f|5f', s):
        return False
    # 允许内层闭壳层（如 4f¹⁴），但不能是价层混合；上面已排除
    return True

# ============================================================
# 分析函数
# ============================================================
def analyze_deltas(deltas, ion_name):
    if len(deltas) < 5:
        return None
    kde = gaussian_kde(deltas)
    x_grid = np.linspace(min(deltas)-0.05, max(deltas)+0.05, 200)
    peak = x_grid[np.argmax(kde(x_grid))]
    print(f"  {ion_name}: 纯 5d 能级 {len(deltas)} 条, δ_peak = {peak:.4f}")

    # 绘图
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(deltas, bins=15, density=True, alpha=0.5, color='darkorange')
    ax.plot(x_grid, kde(x_grid), 'r-', label='KDE')
    ax.set_xlabel('Quantum Defect δ')
    ax.set_ylabel('Density')
    ax.set_title(f'{ion_name} (pure 5d)')
    ax.legend()
    fname = os.path.join(PLOT_DIR, f"{ion_name.replace(' ','_')}_pure_5d.pdf")
    fig.savefig(fname)
    plt.close(fig)
    return peak

# ============================================================
# 主审计
# ============================================================
def main():
    print("=" * 70)
    print("       5d 纯组态量子亏损审计 (排除 6s/6p/4f 混合)")
    print("=" * 70)
    results = []
    for ion in ION_LIST:
        fname = ion["file"]
        if not os.path.exists(fname):
            print(f"  {ion['name']}: 文件 {fname} 不存在")
            continue

        records = read_nist_csv(fname)
        if not records:
            continue

        # 提取电离限
        limit = None
        levels = []
        for r in records:
            if r[0] == 'Limit':
                limit = r[1]
            else:
                levels.append((r[1], r[2]))  # (config, energy)

        if limit is None:
            limit = ion["limit_fb"]
            print(f"  {ion['name']}: 未读取到电离限，使用估计值 {limit} cm⁻¹")
        else:
            print(f"  {ion['name']}: 电离限 = {limit:.0f} cm⁻¹")

        z_eff = ion["z_eff"]

        # 筛选纯 5d 组态并计算量子亏损
        deltas = []
        current_config = ""
        for config, energy in levels:
            if config:
                current_config = config
            if not is_pure_5d(current_config):
                continue
            # 提取主量子数 (通常为5)
            n_match = re.search(r'(\d+)d', current_config)
            if not n_match:
                continue
            n = int(n_match.group(1))
            if n > MAX_N:
                continue
            term = limit - energy
            if term <= 0:
                continue
            n_eff = np.sqrt((RYDBERG * (z_eff**2)) / term)
            delta = n - n_eff
            if 0 <= delta <= n:
                deltas.append(delta)

        if len(deltas) < 5:
            print(f"  {ion['name']}: 纯 5d 能级不足 (仅{len(deltas)}条)")
            continue

        peak = analyze_deltas(deltas, ion["name"])
        if peak is not None:
            results.append((ion["name"], ion["Z"], peak))

    # 汇总表
    print("\n" + "=" * 70)
    print("汇总表 (纯 5d 序列)")
    print("-" * 70)
    for name, Z, peak in results:
        print(f"{name:10s} (Z={Z:2d})  δ_peak = {peak:.4f}")
    print("=" * 70)

    # 阶梯图
    if results:
        zs = [r[1] for r in results]
        peaks = [r[2] for r in results]
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(zs, peaks, 'o--', markersize=8, color='darkorange')
        for i, r in enumerate(results):
            ax.annotate(r[0], (zs[i], peaks[i]), textcoords="offset points", xytext=(0,10), ha='center')
        ax.set_xlabel('Atomic Number Z')
        ax.set_ylabel('Quantum Defect δ peak')
        ax.set_title('5d Series Pure Configurations (no 6s/6p/4f mixing)')
        ax.grid(True, alpha=0.3)
        fname = os.path.join(PLOT_DIR, "pure_5d_ladder.pdf")
        fig.savefig(fname)
        plt.close(fig)
        print(f"阶梯图已保存至 {fname}")

if __name__ == "__main__":
    main()