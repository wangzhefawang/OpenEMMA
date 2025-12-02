# 归一化处理对比：main-original.py vs 当前版本

## 📊 **核心差异总结**

| 方面 | main-original.py | 当前 eval.py（已修复） | 影响 |
|------|------------------|----------------------|------|
| **归一化操作** | `obs_curvatures * 100` | `obs_curvatures * 100` | ✅ 一致 |
| **变量命名** | 直接覆盖 `obs_curvatures` | 新变量 `obs_curvatures_scaled` | ⚠️ 语义更清晰 |
| **格式化精度** | `.1f`（1位小数） | `.3f`（3位小数） | ⚠️ **关键差异** |
| **打印信息** | `Observed Speed and Curvature` | `Observed Speed and Curvature (scaled)` | ✅ 更明确 |
| **连续性约束** | ❌ 无 | ✅ 有（`last_speed`, `last_curvature`） | ✅ 改进 |

---

## 🔍 **详细对比**

### **1. main-original.py（第211-220行）**

```python
# Convert array waypoints to string.
obs_waypoints_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_waypoints]
obs_waypoints_str = ", ".join(obs_waypoints_str)
obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)

obs_curvatures = obs_curvatures * 100  # ← 直接覆盖原变量

obs_speed_curvature_str = [
    f"[{x[0]:.1f},{x[1]:.1f}]"  # ← .1f (1位小数)
    for x in zip(obs_velocities_norm, obs_curvatures)
]
obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

print(f'Observed Speed and Curvature: {obs_speed_curvature_str}')
```

**示例输出**：
```
Observed Speed and Curvature: [0.0,0.0], [0.0,0.0], [0.1,8.1], [0.3,0.6], [0.6,1.9], [0.8,1.0]
                                                      ↑   ↑
                                                速度  曲率（1位小数）
```

---

### **2. 当前 eval.py（已修复，第109-124行）**

```python
# Convert array waypoints to string.
obs_waypoints_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_waypoints]
obs_waypoints_str = ", ".join(obs_waypoints_str)
obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)

# ✅ 修复归一化问题：曲率缩放100倍
obs_curvatures_scaled = obs_curvatures * 100  # ← 使用新变量名

obs_speed_curvature_str = [
    f"[{v:.3f},{k:.3f}]"  # ← .3f (3位小数)
    for v, k in zip(obs_velocities_norm, obs_curvatures_scaled)
]
obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

print(f"\nObserved Speed and Curvature (scaled): {obs_speed_curvature_str}")
```

**示例输出**：
```
Observed Speed and Curvature (scaled): [0.028,0.000], [0.028,0.000], [0.141,8.100], [0.274,0.600], [0.630,1.900], [0.821,1.000]
                                                        ↑     ↑
                                                   速度  曲率（3位小数）
```

---

## ⚠️ **关键差异：格式化精度**

### **差异详解**

| 原始曲率值 | ×100缩放后 | .1f格式 | .3f格式 |
|-----------|-----------|---------|---------|
| 0.0001 | 0.01 | `0.0` | `0.010` |
| 0.0081 | 0.81 | `0.8` | `0.810` |
| 0.0106 | 1.06 | `1.1` | `1.060` |
| 0.0234 | 2.34 | `2.3` | `2.340` |

### **影响分析**

#### **1位小数 (.1f) - 原始版本**

**优点**：
- ✅ 更简洁，数字更短
- ✅ 减少 token 数量（prompt 更短）
- ✅ 可能更符合论文原始实现

**缺点**：
- ❌ **精度损失**：0.81 → 0.8，损失10%信息
- ❌ 小曲率被舍入为0：0.01 → 0.0
- ❌ VLM 看到的信息更粗糙

#### **3位小数 (.3f) - 当前版本**

**优点**：
- ✅ **更高精度**：保留更多细节
- ✅ 小曲率可见：0.010 vs 0.0
- ✅ VLM 可以学习更细微的模式
- ✅ 与速度格式一致（速度也用 .3f）

**缺点**：
- ❌ Prompt 更长（增加 token）
- ❌ 可能过于精细（对于这个任务）

---

## 📈 **实际影响评估**

### **测试案例**

假设观测曲率序列（原始值）：
```
[0.0001, 0.0002, 0.0050, 0.0081, 0.0106, 0.0234, 0.0189, 0.0100]
```

**缩放100倍后**：
```
[0.01, 0.02, 0.50, 0.81, 1.06, 2.34, 1.89, 1.00]
```

**格式化对比**：

| 索引 | 缩放后 | .1f 输出 | .3f 输出 | 信息损失 |
|------|--------|----------|----------|----------|
| 0 | 0.01 | `0.0` | `0.010` | ⚠️ 丢失 |
| 1 | 0.02 | `0.0` | `0.020` | ⚠️ 丢失 |
| 2 | 0.50 | `0.5` | `0.500` | ✅ 保留 |
| 3 | 0.81 | `0.8` | `0.810` | ⚠️ 部分损失 |
| 4 | 1.06 | `1.1` | `1.060` | ⚠️ 四舍五入 |
| 5 | 2.34 | `2.3` | `2.340` | ⚠️ 部分损失 |
| 6 | 1.89 | `1.9` | `1.890` | ⚠️ 部分损失 |
| 7 | 1.00 | `1.0` | `1.000` | ✅ 保留 |

**1位小数版本输出**：
```
[0.0, 0.0, 0.5, 0.8, 1.1, 2.3, 1.9, 1.0]
 ↑    ↑
两个小曲率变成0，趋势丢失
```

**3位小数版本输出**：
```
[0.010, 0.020, 0.500, 0.810, 1.060, 2.340, 1.890, 1.000]
  ↑      ↑
小曲率可见，趋势保留
```

---

## 🎯 **建议：应该使用哪个精度？**

### **方案A：保持3位小数（当前）** ✅ **推荐**

**理由**：
1. ✅ **与速度一致**：速度用 `.3f`，曲率也用 `.3f`，统一
2. ✅ **保留细节**：尤其对于小曲率（<0.1的情况）
3. ✅ **连续性更好**：更精细的数值有助于 VLM 理解平滑过渡
4. ✅ **与当前 prompt 改进一致**：连续性约束需要精确值

**代价**：
- ⚠️ Prompt 稍长（每个数字多2个字符）
- ⚠️ Token 消耗略增（约10-20 tokens）

---

### **方案B：改回1位小数（原始）**

**理由**：
1. ✅ **与论文一致**：如果 main-original.py 是论文实现
2. ✅ **更简洁**：减少 token 消耗
3. ✅ **可能够用**：对于大多数场景，1位小数可能足够

**代价**：
- ❌ 小曲率信息丢失
- ❌ 与速度精度不一致
- ❌ 连续性约束可能不够精确

---

## 💡 **最终建议**

### **保持当前的3位小数 (.3f)** ✅

**原因**：
1. 我们已经添加了**连续性约束**（`The CURRENT speed is X and curvature is Y`），这需要精确的数值
2. 速度使用 `.3f`，曲率也应该保持一致
3. 精度损失可能影响预测质量，尤其是起始点连续性
4. Token 消耗的增加（~10-20）是可接受的

### **如果遇到性能问题，可以考虑折中方案**

#### **方案C：自适应精度** 🔄

```python
# 小曲率用3位，大曲率用1位
obs_speed_curvature_str = [
    f"[{v:.3f},{k:.3f}]" if abs(k) < 1.0 else f"[{v:.3f},{k:.1f}]"
    for v, k in zip(obs_velocities_norm, obs_curvatures_scaled)
]
```

#### **方案D：速度也改为1位小数**

```python
# 两者都用1位小数，保持一致
obs_speed_curvature_str = [
    f"[{v:.1f},{k:.1f}]" for v, k in zip(obs_velocities_norm, obs_curvatures_scaled)
]
```

但这会损失速度信息，不推荐。

---

## 📝 **需要做的决定**

### **选项1：保持当前设置（3位小数）** ✅ **推荐**

**无需修改**，继续测试性能。

### **选项2：改为1位小数（与原版一致）**

如果需要修改：

```python
# eval.py 第120行
obs_speed_curvature_str = [
    f"[{v:.1f},{k:.1f}]" for v, k in zip(obs_velocities_norm, obs_curvatures_scaled)
]
```

### **选项3：A/B测试**

分别测试两种精度，对比 ADE 指标：
- 3位小数版本
- 1位小数版本

看哪个效果更好。

---

## 🎓 **总结**

### **主要发现**

1. ✅ **归一化操作一致**：两个版本都是 `×100`
2. ⚠️ **精度差异**：原版 `.1f` vs 当前 `.3f`
3. ✅ **当前版本改进**：
   - 更清晰的变量命名（`obs_curvatures_scaled`）
   - 更明确的打印信息（`(scaled)`）
   - 添加了连续性约束

### **建议**

**保持当前的3位小数设置**，因为：
- 与速度精度一致
- 更好地支持连续性约束
- 保留更多信息给 VLM

如果后续发现性能问题或想与原版严格一致，可以考虑改回1位小数。

---

**更新日期**: 2025-12-02  
**对比对象**: main-original.py vs eval.py (已修复)  
**结论**: 归一化正确，精度略有差异（建议保持当前）

