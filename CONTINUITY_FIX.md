# 预测连续性问题修复文档

## 📋 问题描述

### 问题现象

在实际运行中发现，VLM 预测的未来速度/曲率与观测数据**缺乏连续性**：

**示例：**
```
观测数据（历史5秒，最后10个时间步）：
[0.028,0.000], [0.028,0.000], [0.003,0.000], [0.002,0.000], [0.001,0.000], 
[0.005,0.000], [0.141,0.081], [0.274,0.006], [0.630,0.019], [0.821,0.010]
                                                              ↑
                                                         当前速度：0.821 m/s
                                                         车辆正在加速！

VLM预测（未来5秒，接下来10个时间步）：
[0.033,0.000], [0.032,0.000], [0.031,0.000], [0.030,0.000], [0.029,0.000], 
[0.028,0.000], [0.027,0.000], [0.026,0.000], [0.025,0.000], [0.024,0.000]
 ↑
预测起始速度：0.033 m/s
完全断裂！
```

### 问题分析

1. **物理不合理**：
   - 车辆不可能在 0.5 秒内从 0.821 m/s 骤降到 0.033 m/s
   - 这相当于减速 **96%**，需要极端刹车（约 -1.6 m/s²）

2. **趋势断裂**：
   - 观测数据显示车辆正在**加速**（0.001 → 0.821）
   - 预测却从一个很小的速度开始**减速**（0.033 → 0.024）

3. **根本原因**：
   - Prompt 没有明确强调**时间连续性**
   - 模型可能把"历史数据"和"未来预测"当作两个独立任务
   - 缺少明确的"从当前状态继续"的指令

## 🔧 解决方案

### 改进策略

在 `eval.py` 的 `GenerateMotion()` 函数中修改 prompt，增加以下要素：

1. **明确当前状态**：提取并显式告知最后一个速度和曲率
2. **强调连续性**：要求预测"从当前状态开始"并"平滑过渡"
3. **约束起点**：指定"第一个预测速度应该接近当前速度"

### 修改前（问题版本）

```python
prompt = f"""...
The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
Based on the historical data and scene context, predict the NEXT 10 future timesteps...
"""
```

**问题**：
- ❌ 没有提到"从最后一个速度继续"
- ❌ 没有强调"平滑过渡"
- ❌ 模型可能忽略最后一个状态

### 修改后（改进版本）

```python
# 提取最后一个观测速度和曲率
last_speed = obs_velocities_norm[-1]
last_curvature = obs_curvatures[-1]

prompt = f"""...
The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
The CURRENT speed is {last_speed:.3f} and curvature is {last_curvature:.3f}. 
Your predictions must START from this current state and smoothly continue the motion. 
Predict the NEXT 10 future timesteps (5 seconds ahead) considering the car's momentum and intent. 
The first predicted speed should be close to {last_speed:.3f}. 
Only output the future predictions, do not repeat the historical data...
"""
```

**改进点**：
- ✅ **显式声明当前状态**：`The CURRENT speed is {last_speed:.3f}`
- ✅ **强调连续性约束**：`must START from this current state and smoothly continue`
- ✅ **提供起点参考**：`The first predicted speed should be close to {last_speed:.3f}`
- ✅ **考虑物理约束**：`considering the car's momentum`

## 📊 预期效果

### 修改前（断裂）

```
观测最后：[0.821, 0.010]
         ↓ 断裂！速度骤降 96%
预测开始：[0.033, 0.000]
```

### 修改后（连续）

```
观测最后：[0.821, 0.010]
         ↓ 平滑过渡
预测开始：[0.850, 0.012]  ← 接近 0.821，合理延续
```

## 🎯 影响范围

### 直接影响

1. **预测质量提升**：
   - 更符合物理规律
   - 更好的时间连续性
   - 减少突变预测

2. **ADE 指标改善**：
   - 尤其是 ADE-1s（1秒预测）
   - 起点连续性改善会显著降低初始误差

3. **运动规划合理性**：
   - 预测轨迹更平滑
   - 符合车辆动力学约束

### 适用场景

所有使用 OpenEMMA 进行运动预测的场景：
- ✅ 加速场景（避免预测突然减速）
- ✅ 减速场景（避免预测突然加速）
- ✅ 转向场景（保持曲率连续性）
- ✅ 匀速场景（保持速度稳定性）

## 🧪 测试建议

### 测试用例

1. **加速场景**：
   - 观测：0.1 → 0.5 → 1.0 m/s
   - 预期：预测从 ≈1.0 m/s 开始

2. **急刹车场景**：
   - 观测：1.5 → 1.2 → 0.8 m/s
   - 预期：预测从 ≈0.8 m/s 开始，继续减速

3. **转向场景**：
   - 观测：k = 0.001 → 0.005 → 0.015
   - 预期：预测从 ≈0.015 开始

### 验证方法

```python
# 在 main.py 中添加连续性检查
last_obs_speed = obs_velocities_norm[-1]
first_pred_speed = speed_curvature_pred[0][0]

continuity_error = abs(first_pred_speed - last_obs_speed)
print(f"连续性误差: {continuity_error:.3f} m/s")

if continuity_error > 0.5:  # 0.5 m/s 阈值
    print(f"⚠️ 警告：预测缺乏连续性！")
```

## 📝 相关文件

- `eval.py` - 包含改进的 prompt
- `main.py` - 主推理流程
- `FIX_PREDICTION_PARSING.md` - 预测解析问题修复
- `PROMPT_COMPARISON.md` - Prompt 对比文档

## 🔮 未来改进

### 可能的进一步优化

1. **物理约束强化**：
   ```python
   max_accel = 3.0  # m/s²
   max_decel = -5.0  # m/s²
   prompt += f"Maximum acceleration is {max_accel} m/s², maximum deceleration is {max_decel} m/s²."
   ```

2. **多步连续性**：
   - 不仅约束第一个预测点
   - 要求整个预测序列平滑

3. **动态约束**：
   - 根据当前速度自适应调整容忍度
   - 高速时允许更大的变化范围

4. **后处理平滑**：
   ```python
   # 如果连续性仍不满意，后处理强制平滑
   if abs(pred[0] - obs[-1]) > threshold:
       pred[0] = 0.7 * obs[-1] + 0.3 * pred[0]  # 加权平滑
   ```

## ✅ 检查清单

部署前检查：
- [x] 修改了 `eval.py` 中的 prompt
- [x] 两种模式（openemma 和其他）都已更新
- [x] 语法检查通过（无 linter 错误）
- [ ] 在测试场景上验证连续性改善
- [ ] 对比修改前后的 ADE 指标
- [ ] 检查是否引入新问题（如模型不理解新指令）

## 📚 参考

- **物理约束**：车辆动力学一般限制加速度在 [-8, 4] m/s² 范围
- **时间间隔**：nuScenes 采样率为 2 Hz（0.5 秒间隔）
- **平滑性要求**：相邻时间步速度变化一般不超过 2 m/s

---

**更新日期**: 2025-12-02  
**相关 Issue**: 预测连续性问题（用户反馈）  
**影响版本**: main.py (重构版)

