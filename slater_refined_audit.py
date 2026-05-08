#!/usr/bin/env python3
"""
slater_refined_audit_debug.py
带调试输出的 Slater 修正版审计脚本。
"""
import csv
import numpy as np
import re
import os
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# ============================================================
# 1. 离子配置
# ============================================================
ION_3D = [
    {"name": "Ti II", "Z": 22, "n_shell": 3, "d_electrons": 2, "file": "TiII.csv", "limit_fb": 109494},
    {"name": "V II",  "Z": 23, "n_shell": 3, "d_electrons": 3, "file": "VII.csv",  "limit_fb": 118030},
    {"name": "Cr II", "Z": 24, "n_shell": 3, "d_electrons": 4, "file": "CrII.csv", "limit_fb": 132971},
    {"name": "Fe II", "Z": 26, "n_shell": 3, "d_electrons": 6, "file": "FeII.csv", "limit_fb": 130655},
    {"name": "Cu II", "Z": 29, "n_shell": 3, "d_electrons": 9, "file": "CuII.csv", "limit_fb": 163669},
]

ION_4D = [
    {"name": "Y III",  "Z": 39, "n_shell": 4, "d_electrons": 1, "file": "YIII.csv",  "limit_fb": 165540},
    {"name": "Zr III", "Z": 40, "n_shell": 4, "d_electrons": 2, "file": "ZrIII.csv", "limit_fb": 186880},
    {"name": "Nb III", "Z": 41, "n_shell": 4, "d_electrons": 3, "file": "NbIII.csv", "limit_fb": 202000},
    {"name": "Mo III", "Z": 42, "n_shell": 4, "d_electrons": 4, "file": "MoIII.csv", "limit_fb": 218800},
    {"name": "Ru III", "Z": 44, "n_shell": 4, "d_electrons": 6, "file": "RuIII.csv", "limit_fb": 229600},
    {"name": "Rh III", "Z": 45, "n_shell": 4, "d_electrons": 7, "file": "RhIII.csv", "limit_fb": 250500},
    {"name": "Pd III", "Z": 46, "n_shell": 4, "d_electrons": 8, "file": "PdIII.csv", "limit_fb": 265600},
    {"name": "Ag III", "Z": 47, "n_shell": 4, "d_electrons": 9, "file": "AgIII.csv", "limit_fb": 280900},
]

ION_5D = [
    {"name": "Lu II",  "Z": 71, "n_shell": 5, "d_electrons": 1, "file": "LuII.csv",  "limit_fb": 113970},
    {"name": "Hf II",  "Z": 72, "n_shell": 5, "d_electrons": 2, "file": "HfII.csv",  "limit_fb": 117820},
    {"name": "Ta II",  "Z": 73, "n_shell": 5, "d_electrons": 3, "file": "TaII.csv",  "limit_fb": 131000},
    {"name": "W II",   "Z": 74, "n_shell": 5, "d_electrons": 4, "file": "WII.csv",   "limit_fb": 139000},
    {"name": "Os III", "Z": 76, "n_shell": 5, "d_electrons": 6, "file": "OsIII.csv", "limit_fb": 202000},
]

RYDBERG = 109737.3156
MAX_N = 10

# ============================================================
# 2. Slater规则
# ============================================================
def calculate_slater_z_star(Z, n_shell, d_electrons):
    if n_shell == 3:
        inner = 18
    elif n_shell == 4:
        inner = 36
    elif n_shell == 5:
        inner = 68
    else:
        inner = 0
    same_group = (d_electrons - 1) * 0.35
    S = inner + same_group
    return Z - S

# ============================================================
# 3. CSV读取
# ============================================================
def read_nist_csv(filepath):
    if not os.path.exists(filepath):
        return None
    records = []
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
            if 'limit' in config.lower() or 'limit' in term.lower():
                clean = re.sub(r'[^\d.]', '', energy_raw.replace(' ', ''))
                if clean: records.append(('Limit', float(clean)))
                continue
            clean_e = re.sub(r'[^\d.]', '', energy_raw.replace(' ', '').replace('[', '').replace(']', ''))
            if clean_e: records.append(('Level', config, float(clean_e)))
    return records

def is_pure_d(config, series):
    s = config.lower()
    if f'{series}d' not in s: return False
    if series == 3 and re.search(r'3d\.4[sp]', s): return False
    elif series == 4 and re.search(r'4d\.5[sp]', s): return False
    elif series == 5:
        if re.search(r'5d\.6[sp]', s): return False
        if re.search(r'4f|5f', s): return False
    return True

def get_kde_peak(deltas):
    if len(deltas) < 5: return None
    kde = gaussian_kde(deltas)
    x_grid = np.linspace(min(deltas)-0.05, max(deltas)+0.05, 200)
    return x_grid[np.argmax(kde(x_grid))]

# ============================================================
# 4. 审计序列（带调试输出）
# ============================================================
def audit_series_slater(ion_list, series):
    results = []
    for ion in ion_list:
        fname = ion["file"]
        print(f"  检查文件: {fname} ... ", end="")
        if not os.path.exists(fname):
            print("不存在")
            continue
        else:
            print("找到")

        records = read_nist_csv(fname)
        if not records:
            print(f"    警告: {fname} 无有效记录")
            continue

        levels, limit = [], None
        for r in records:
            if r[0] == 'Limit':
                limit = r[1]
            else:
                levels.append((r[1], r[2]))
        if limit is None:
            limit = ion["limit_fb"]
            print(f"    使用估计电离限 {limit}")

        Z_star = calculate_slater_z_star(ion["Z"], ion["n_shell"], ion["d_electrons"])
        print(f"    Z* = {Z_star:.2f}")

        deltas = []
        current_config = ""
        for config, energy in levels:
            if config: current_config = config
            if not is_pure_d(current_config, series): continue
            n_match = re.search(r'(\d+)d', current_config)
            if not n_match: continue
            n = int(n_match.group(1))
            if n > MAX_N: continue
            term = limit - energy
            if term <= 0: continue
            n_eff = np.sqrt((RYDBERG * (Z_star**2)) / term)
            delta = n - n_eff
            if 0 <= delta <= n: deltas.append(delta)

        print(f"    纯d能级数: {len(deltas)}")
        if len(deltas) < 5:
            print(f"    跳过 (不足5条)")
            continue

        peak = get_kde_peak(deltas)
        if peak is None:
            print("    KDE峰提取失败")
            continue

        print(f"    δ_peak = {peak:.4f}")
        results.append((ion["name"], ion["Z"], peak, series, Z_star, len(deltas)))
    return results

# ============================================================
# 5. 主程序
# ============================================================
def main():
    print("Slater Z* 审计开始...")
    print("="*60)
    all_results = []
    all_results.extend(audit_series_slater(ION_3D, 3))
    all_results.extend(audit_series_slater(ION_4D, 4))
    all_results.extend(audit_series_slater(ION_5D, 5))

    if not all_results:
        print("未找到任何可分析的数据！请确保 CSV 文件与脚本在同一目录。")
        return

    print("\n" + "="*60)
    print(f"{'Ion':8s} {'Z':3s} {'Ser':4s} {'Z*':6s} {'N':5s} {'δ_peak':8s}")
    print("-"*40)
    for r in all_results:
        print(f"{r[0]:8s} {r[1]:3d} {r[3]}d   {r[4]:6.2f} {r[5]:5d} {r[2]:8.4f}")

    # 蒙特卡洛显著性检验
    if all_results:
        peaks = [r[2] for r in all_results]
        labels = [r[3] for r in all_results]
        
        # 模拟: 如果 δ 是均匀随机的，三个非重叠聚类的概率
        np.random.seed(42)
        n_trials = 50000
        count = 0
        target_means = {3: 1.0, 4: np.sqrt(3), 5: np.sqrt(8)}
        tolerance = 0.1  # 10% 容差

        for _ in range(n_trials):
            sim = np.random.uniform(0, 4, len(peaks))
            match = True
            for s in [3,4,5]:
                cluster = sim[[i for i, sl in enumerate(labels) if sl == s]]
                if len(cluster) > 0:
                    mean_val = np.mean(cluster)
                    target = target_means[s]
                    if abs(mean_val - target) > tolerance * target:
                        match = False
                        break
            if match: count += 1
        p_val = count / n_trials
        print(f"\n蒙特卡洛模拟 (N={n_trials})")
        print(f"随机形成三个几何聚类的概率 p = {p_val:.6f}")
        if p_val < 0.001:
            print("结论: 在 99.9% 置信水平上拒绝连续空间零假设。")
        elif p_val < 0.01:
            print("结论: 在 99% 置信水平上拒绝连续空间零假设。")

if __name__ == "__main__":
    main()