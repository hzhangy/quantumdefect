import numpy as np

# ============================================================
# 1. 原始审计账本 (NIST 筛选出的 KDE 峰值)
# ============================================================
# 3d 系列 (Ti II, V II, Cr II, Fe II, Cu II)
d3 = np.array([0.8443, 0.8601, 0.9035, 0.9721, 0.7984])
# 4d 系列 (Zr III, Nb III, Mo III, Rh III, Pd III) - 纠正后
d4 = np.array([1.7286, 1.7954, 1.9167, 1.9523, 2.0328])
# 5d 系列 (Lu II, Hf II, Ta II, W II, Os III)
d5 = np.array([2.6267, 2.7710, 3.0779, 3.1470, 2.6691])

# 理论几何目标
TARGETS = np.array([1.0, np.sqrt(3), np.sqrt(8)])

def run_precision_audit():
    # 2. 计算实测均值
    m1, m2, m3 = np.mean(d3), np.mean(d4), np.mean(d5)
    observed_means = np.array([m1, m2, m3])
    
    # 3. 计算实测总偏差 (L1 范数)
    observed_error = np.sum(np.abs(observed_means - TARGETS))
    
    print("="*60)
    print(f"{'N.E.A. 几何对齐显著性审计 (V2.0)':^60}")
    print("="*60)
    print(f"  3d 实测均值: {m1:.4f} (目标 1.000, 偏差: {m1-1.0:.4f})")
    print(f"  4d 实测均值: {m2:.4f} (目标 1.732, 偏差: {m2-1.732:.4f})")
    print(f"  5d 实测均值: {m3:.4f} (目标 2.828, 偏差: {m3-2.828:.4f})")
    print(f"  总账目偏差 (Total L1 Error): {observed_error:.4f}")
    print("-" * 60)

    # 4. 蒙特卡洛模拟：随机点“撞大运”能撞多准？
    np.random.seed(42)
    trials = 1000000 # 进行一百万次模拟
    hit_count = 0
    
    # 模拟背景：在 0 到 3.5 的物理允许范围内随机撒 15 个点
    for _ in range(trials):
        # 随机产生 15 个点
        sim = np.random.uniform(0, 3.5, 15)
        sim.sort()
        
        # 将随机点分成三组（模拟三个周期）
        sm1 = np.mean(sim[:5])
        sm2 = np.mean(sim[5:10])
        sm3 = np.mean(sim[10:])
        
        # 计算随机模拟的总偏差
        sim_error = np.sum(np.abs(np.array([sm1, sm2, sm3]) - TARGETS))
        
        # 如果随机偏差比实测偏差还要小，算作随机成功
        if sim_error < observed_error:
            hit_count += 1
            
    p_value = hit_count / trials
    
    print(f"  模拟次数: {trials}")
    print(f"  随机命中次数: {hit_count}")
    print(f"  显著性 p-value: {p_value:.6f}")
    
    # 5. 审计结论
    print("-" * 60)
    if p_value < 0.001:
        print("  结论：[极度 Solid] 拒绝零假设。")
        print("  物理意义：能级残差向几何常数的聚类具有强制性的硬件来源。")
    elif p_value < 0.05:
        print("  结论：[统计显著] 空间离散性具有很强的实验支撑。")
    else:
        print("  结论：[存在疑点] 目前的对齐精度尚不足以判定为非随机。")
    print("=" * 60)

if __name__ == "__main__":
    run_precision_audit()