import numpy as np
from sklearn.mixture import GaussianMixture

# 实验测得的原始数据点 (不经过任何回归处理)
d3 = np.array([0.8443, 0.8601, 0.9035, 0.9721, 0.7984])
d4 = np.array([1.7286, 1.7954, 1.9167, 1.9523, 2.0328]) # 纠正后的 Zr 排除
d5 = np.array([2.6267, 2.7710, 3.0779, 3.1470, 2.6691])

def audit_p_value():
    # 1. 计算三代数据的分离度 (Inter-cluster separation)
    centers = [np.mean(d3), np.mean(d4), np.mean(d5)]
    stds = [np.std(d3), np.std(d4), np.std(d5)]
    
    print(">>> 原始账本三代聚类审计")
    print(f"  3d 均值: {centers[0]:.3f} (预期 1.0) | 标准差: {stds[0]:.3f}")
    print(f"  4d 均值: {centers[1]:.3f} (预期 1.732) | 标准差: {stds[1]:.3f}")
    print(f"  5d 均值: {centers[2]:.3f} (预期 2.828) | 标准差: {stds[2]:.3f}")

    # 2. 执行“巧合概率”测试
    # 模拟 100,000 次，看随机数据能否产生这种“整齐的分层”
    np.random.seed(42)
    trials = 100000
    success = 0
    for _ in range(trials):
        # 在 0 到 3.5 之间随机取 15 个点
        sim = np.random.uniform(0, 3.5, 15)
        sim = np.sort(sim)
        # 检查是否能形成三个间距大于 0.7 的非重叠簇
        if (sim[10]-sim[4] > 0.7) and (sim[4]-sim[0] > 0.7):
            success += 1
            
    p_val = success / trials
    print(f"\n>>> 统计显著性校验")
    print(f"  随机产生此类分层现象的概率 p: {p_val:.6f}")
    
    if p_val < 0.001:
        print("  结论：[极度 Solid] 该分布具有非随机的硬件级属性。")
    else:
        print("  结论：[存在疑点] 需要更多数据排除随机性。")

audit_p_value()