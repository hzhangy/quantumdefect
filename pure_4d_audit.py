#!/usr/bin/env python3
"""
pure_4d_audit.py — 纯 4d 组态量子亏损审计
仅提取组态为纯 nd (如 4d², 4d³) 的能级，
忽略混合了 5s, 5p 等的价层激发态，
以消除外层电子对量子亏损的干扰。
"""

import csv
import numpy as np
import re
import os
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# ============================================================
# 离子配置 (可根据已下载的 CSV 调整)
# ============================================================
ION_LIST = [
    {"name": "Y III",  "Z": 39, "file": "YIII.csv",  "z_eff": 3.0, "limit_fb": 165540.5},
    {"name": "Zr III", "Z": 40, "file": "ZrIII.csv", "z_eff": 3.0, "limit_fb": 186880.0},
    {"name": "Nb III", "Z": 41, "file": "NbIII.csv", "z_eff": 3.0, "limit_fb": 202000.0},
    {"name": "Mo III", "Z": 42, "file": "MoIII.csv", "z_eff": 3.0, "limit_fb": 218800.0},
    {"name": "Ru III", "Z": 44, "file": "RuIII.csv", "z_eff": 3.0, "limit_fb": 229600.0},
    {"name": "Rh III", "Z": 45, "file": "RhIII.csv", "z_eff": 3.0, "limit_fb": 250500.0},
    {"name": "Pd III", "Z": 46, "file": "PdIII.csv", "z_eff": 3.0, "limit_fb": 265600.0},
    {"name": "Ag III", "Z": 47, "file": "AgIII.csv", "z_eff": 3.0, "limit_fb": 280900.0},
]

RYDBERG = 109737.3156
MAX_N = 10
PLOT_DIR = "plots_pure_4d"
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# 读取 NIST CSV (与之前相同的健壮版本)
# ============================================================
def read_nist_csv(filepath):
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            # 自动定位列
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
                if len(row) <= max(config_idx, term_idx, energy_idx):
                    continue
                config = row[config_idx].strip()
                term = row[term_idx].strip()
                energy_raw = row[energy_idx]

                # 检测 Limit 行
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
# 纯 d 组态过滤器
# ============================================================
def is_pure_d(config):
    """仅保留 d 后不紧跟 .5s, .5p 的组态"""
    s = config.lower()
    if 'd' not in s:
        return False
    # 检查 "d.5s" 或 "d.5p" 模式 (价层混合)
    if re.search(r'd\.5[sp]', s):
        return False
    return True

# ============================================================
# 分析函数
# ============================================================
def analyze_pure_deltas(deltas, ion_name):
    if len(deltas) < 5:
        return None
    kde = gaussian_kde(deltas)
    x_grid = np.linspace(min(deltas)-0.05, max(deltas)+0.05, 200)
    peak = x_grid[np.argmax(kde(x_grid))]
    print(f"  {ion_name}: 纯 d 能级 {len(deltas)} 条, 峰值 δ = {peak:.4f}")

    # 简单直方图
    plt.figure(figsize=(6,4))
    plt.hist(deltas, bins=15, density=True, alpha=0.5, color='steelblue')
    plt.plot(x_grid, kde(x_grid), 'r-', label='KDE')
    plt.xlabel('Quantum Defect δ')
    plt.ylabel('Density')
    plt.title(f'{ion_name} (pure 4d)')
    plt.legend()
    fname = os.path.join(PLOT_DIR, f"{ion_name.replace(' ','_')}_pure_4d.pdf")
    plt.savefig(fname)
    plt.close()
    return peak

# ============================================================
# 主审计
# ============================================================
def main():
    print("=" * 70)
    print("         纯 4d 组态量子亏损审计 (排除 5s/5p 混合)")
    print("=" * 70)
    results = []
    for ion in ION_LIST:
        fname = ion["file"]
        if not os.path.exists(fname):
            print(f"  {ion['name']}: 文件 {fname} 不存在，跳过")
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
        z_eff = ion["z_eff"]

        # 过滤纯 d 组态并计算量子亏损
        deltas = []
        current_config = ""
        for config, energy in levels:
            # 处理配置缺失（沿用上一行的配置）
            if config:
                current_config = config
            if not is_pure_d(current_config):
                continue
            # 提取主量子数
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
            print(f"  {ion['name']}: 纯 d 能级不足 (仅{len(deltas)}条)，跳过")
            continue

        peak = analyze_pure_deltas(deltas, ion["name"])
        if peak is not None:
            results.append((ion["name"], ion["Z"], peak))

    # 打印汇总
    print("\n" + "=" * 70)
    print("汇总表")
    print("-" * 70)
    for name, Z, peak in results:
        print(f"{name:10s} (Z={Z:2d})  δ_peak = {peak:.4f}")
    print("=" * 70)

    # 生成阶梯图
    if results:
        zs = [r[1] for r in results]
        peaks = [r[2] for r in results]
        plt.figure(figsize=(10,5))
        plt.plot(zs, peaks, 'o--', markersize=8)
        for i, r in enumerate(results):
            plt.annotate(r[0], (zs[i], peaks[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Z')
        plt.ylabel('Quantum Defect δ peak')
        plt.title('Pure 4d Sequence (no 5s/5p mixing)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PLOT_DIR, "pure_4d_ladder.pdf"))
        plt.show()

if __name__ == "__main__":
    main()