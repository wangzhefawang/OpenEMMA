# 归一化不一致问题分析

## 🚨 **问题发现**

用户敏锐地发现：**预测结果与观测数据缺乏连续性，可能是归一化导致的。**

## 📊 **数据流对比**

### **main-original.py（原始版本）**

```python
# ========== 步骤1：计算原始曲率 ==========
ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
# 值范围：约 -0.1 ~ 0.1 (1/米)

# ========== 步骤2：提取观测数据并传给 GenerateMotion ==========
obs_ego_curvatures = ego_curvatures[i:i+OBS_LEN]
# 例如：[0.0001, 0.0002, ..., 0.010]

# ========== 步骤3：GenerateMotion 函数内部 - 缩放100倍 ==========
# main-original.py 第215行：
obs_curvatures = obs_curvatures * 100  # ✅ 放大100倍
# 变成：[0.01, 0.02, ..., 1.0]

obs_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(..., obs_curvatures)]
# 输出给 VLM：... [0.8, 1.0], [0.9, 1.2] ...
#                      ↑    ↑
#                   速度  曲率（已缩放）

# ========== 步骤4：VLM 预测（同样尺度） ==========
# VLM 输出：[0.9, 1.1], [1.0, 1.3], ...
#               ↑    ↑
#            速度  曲率（放大100倍的尺度）

# ========== 步骤5：后处理 - 缩回原始尺度 ==========
# main-original.py 第444行：
pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100  # ✅ 除以100
# 变成：[0.011, 0.013, ...] (恢复原始尺度)
```

---

### **main.py + eval.py（当前重构版）**

```python
# ========== 步骤1：计算原始曲率 ==========
ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
# 值范围：约 -0.1 ~ 0.1 (1/米)

# ========== 步骤2：提取观测数据并传给 GenerateMotion ==========
obs_ego_curvatures = ego_curvatures[i:i+OBS_LEN]
# 例如：[0.0001, 0.0002, ..., 0.010]

# ========== 步骤3：GenerateMotion 函数内部 - ❌ 没有缩放 ==========
# eval.py 第112-116行：
obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
obs_speed_curvature_str = [
    f"[{v:.3f},{k:.3f}]" for v, k in zip(obs_velocities_norm, obs_curvatures)
]
# ❌ obs_curvatures 直接使用，没有 * 100
# 输出给 VLM：... [0.800, 0.010], [0.900, 0.012] ...
#                      ↑      ↑
#                   速度   曲率（原始尺度，太小！）

# ========== 步骤4：VLM 预测（数值尺度不一致） ==========
# VLM 看到的曲率是 0.010，但它不知道这个数值的物理意义
# VLM 可能输出：[0.033, 0.000], [0.032, 0.000], ...
#                  ↑      ↑
#               速度   曲率（尺度混乱）

# ========== 步骤5：后处理 - 依然除以100 ==========
# main.py 第314行：
pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100  # ⚠️ 又除以100
# 如果 VLM 输出的是原始尺度 0.001，再除以100变成 0.00001（太小！）
```

---

## ❌ **问题根源**

### **归一化不一致导致的混乱**

| 阶段 | 原始版本 | 当前版本 | 问题 |
|------|----------|----------|------|
| **喂给 VLM** | 曲率 × 100 | 曲率（原始） | ❌ 尺度不一致 |
| **VLM 预测** | 曲率 × 100 | 曲率（？） | ❌ VLM 不知道用什么尺度 |
| **后处理** | 曲率 ÷ 100 | 曲率 ÷ 100 | ❌ 双重错误 |

### **实际影响**

以您终端输出的例子：

```
观测数据（当前版本输出）：
[0.028,0.000], ..., [0.821,0.010]
                          ↑ 曲率 = 0.010

传给 VLM 的 prompt：
"historical velocities and curvatures ... [0.821, 0.010]"
                                                    ↑ 太小！

VLM 可能困惑：
- 速度是 0.821（正常数值）
- 曲率是 0.010（相比速度小100倍，难以理解）

VLM 预测（尺度混乱）：
[0.033, 0.000], [0.032, 0.000], ...
 ↑ 速度断裂（可能因为曲率尺度混乱导致整体预测失败）
```

---

## ✅ **解决方案**

### **方案1：恢复原始归一化（推荐）**

在 `eval.py` 中添加曲率缩放：

```python
# eval.py 第112行之后
obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)

# ✅ 添加曲率归一化
obs_curvatures_scaled = obs_curvatures * 100  # 缩放到更友好的数值范围

obs_speed_curvature_str = [
    f"[{v:.3f},{k:.3f}]" for v, k in zip(obs_velocities_norm, obs_curvatures_scaled)
]

# 同时更新 last_curvature
last_curvature = obs_curvatures_scaled[-1]
```

**原理**：
- 曲率的物理单位是 1/米，典型值在 -0.1 ~ 0.1
- 缩放 100 倍后变成 -10 ~ 10，与速度（0 ~ 20 m/s）数值范围更接近
- VLM 更容易理解这两个数值的关系

---

### **方案2：移除后处理中的除法**

如果决定不缩放输入，那么输出也不应该除以100：

```python
# main.py 第314行
pred_curvatures = np.array(speed_curvature_pred)[:, 1]  # ❌ 不除以100
```

**问题**：这需要 VLM 输出非常小的曲率值（0.001级别），不太直观。

---

## 🎯 **推荐实施方案1**

### **修改文件：eval.py**

```python
# 第112行之后添加
obs_curvatures_scaled = obs_curvatures * 100

# 第114行修改
obs_speed_curvature_str = [
    f"[{v:.3f},{k:.3f}]" for v, k in zip(obs_velocities_norm, obs_curvatures_scaled)
]

# 第125和135行修改
last_curvature = obs_curvatures_scaled[-1]  # 使用缩放后的值
```

---

## 📈 **预期改进**

### **修改前（当前）**

```
Prompt: "historical curvatures ... [0.821, 0.010]"
                                             ↑ 太小，VLM 困惑

VLM 输出: [0.033, 0.000], ... (断裂)
```

### **修改后（方案1）**

```
Prompt: "historical curvatures ... [0.821, 1.000]"
                                             ↑ 合理范围，VLM 理解

VLM 输出: [0.850, 1.100], ... (连续且合理)
          后处理：÷100 → [0.850, 0.011]
```

---

## 🔍 **为什么原始版本使用 ×100？**

### **数值范围对比**

| 物理量 | 原始范围 | 缩放后 | 原因 |
|--------|----------|--------|------|
| **速度** | 0 ~ 20 m/s | 不变 | 已经是合理范围 |
| **曲率** | -0.1 ~ 0.1 (1/m) | -10 ~ 10 | ✅ 与速度同级别 |

### **VLM 的数值理解**

- **速度 0.8 vs 曲率 0.01**：VLM 难以建立关系（相差80倍）
- **速度 0.8 vs 曲率 1.0**：VLM 容易理解（同级别数值）

---

## 📝 **检查清单**

实施方案1后需要验证：

- [ ] eval.py 中添加 `obs_curvatures * 100`
- [ ] 确保 `last_curvature` 使用缩放后的值
- [ ] 保持 main.py 中 `/ 100` 的后处理
- [ ] 测试预测连续性是否改善
- [ ] 检查曲率数值是否在合理范围（-10 ~ 10）

---

## 🎓 **总结**

用户的洞察非常准确！**归一化不一致**确实是导致预测连续性问题的关键原因之一：

1. ✅ **原始版本**：输入 ×100，输出 ÷100（一致）
2. ❌ **当前版本**：输入不缩放，输出 ÷100（不一致）
3. 🎯 **解决方案**：恢复输入的 ×100 缩放

这个问题结合了：
- 数值归一化
- VLM 的数值理解能力
- 物理单位的合理表示

修复后应该能显著改善预测的连续性和准确性！

