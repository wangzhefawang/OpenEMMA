# 🔬 ADE 对比实验设置说明

## 📋 版本说明

### **版本 A：包含所有优化（备份在 Backup_WithFixes）**
- ✅ 预测连续性修复（归一化 + Prompt 改进）
- ✅ 预测解析优化（智能解析）
- ✅ 其他所有优化（CUDA Graphs、显存管理等）

### **版本 B：回退关键改动（当前 main.py + eval.py）**
- ❌ 已回退预测连续性修复
- ❌ 已回退预测解析优化
- ✅ 保留其他优化（CUDA Graphs、显存管理等）

---

## 🔄 已回退的改动详情

### **1. eval.py 的回退（预测连续性修复）**

#### **归一化处理**
```python
# 版本 A（已备份）- 有修复
obs_curvatures_scaled = obs_curvatures * 100
obs_speed_curvature_str = [
    f"[{v:.3f},{k:.3f}]" for v, k in zip(obs_velocities_norm, obs_curvatures_scaled)
]

# 版本 B（当前）- 已回退
obs_curvatures = obs_curvatures * 100  # 直接修改原变量
obs_speed_curvature_str = [
    f"[{v:.1f},{k:.1f}]" for v, k in zip(obs_velocities_norm, obs_curvatures)
]
```

#### **Prompt 连续性约束**
```python
# 版本 A（已备份）- 有修复
prompt = f"""...
The 5 second historical velocities and curvatures are {obs_speed_curvature_str}. 
The CURRENT speed is {last_speed:.3f} and curvature is {last_curvature:.3f}.
Your predictions must START from this current state and smoothly continue the motion.
The first predicted speed should be close to {last_speed:.3f}.
Only output the future predictions, do not repeat the historical data.
..."""

# 版本 B（当前）- 已回退
prompt = f"""...
The 5 second historical velocities and curvatures are {obs_speed_curvature_str}. 
Infer the association between these numbers and the image sequence.
..."""
```

---

### **2. main.py 的回退（预测解析优化）**

```python
# 版本 A（已备份）- 智能解析
# 策略1：找到最后一次出现的 "future speeds and curvatures:"
last_marker_pos = pred_text.rfind("future speeds and curvatures:")
if last_marker_pos != -1:
    pred_waypoints = prediction[last_marker_pos + len("..."):]
# 策略2：跳过观测数据部分
elif obs_marker_pos != -1:
    pred_waypoints = prediction[obs_marker_pos:]
    # 移除包含观测数据的句子
# 策略3：向后兼容
else:
    pred_waypoints = prediction

# 版本 B（当前）- 简单替换
pred_waypoints = prediction.replace("Future speeds and curvatures:", "").strip()
coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints)
```

---

## 🧪 对比实验步骤

### **步骤 1：运行版本 B（当前版本，已回退改动）**

```bash
# 测试场景
python main.py \
    --model-path "D:\SAVE\files\Models\Qwen2.5-VL-7B-Instruct" \
    --scenes scene-0103,scene-1077 \
    --quantization none \
    --use-cuda-graphs \
    --dataroot "D:\SAVE\files\Datasets\nuscenes-v1.0" \
    --version v1.0-trainval

# 结果保存在：[model]_results/openemma/[timestamp]/
# 记录输出目录名称，例如：20250102-030000
```

---

### **步骤 2：恢复版本 A（包含所有修复）**

```bash
# 恢复带修复的版本
cd "D:\SAVE\files\Github\OpenEMMA"
Copy-Item Backup_WithFixes\eval.py eval.py -Force
Copy-Item Backup_WithFixes\main.py main.py -Force
```

---

### **步骤 3：运行版本 A（包含所有优化）**

```bash
# 使用相同参数运行
python main.py \
    --model-path "D:\SAVE\files\Models\Qwen2.5-VL-7B-Instruct" \
    --scenes scene-0103,scene-1077 \
    --quantization none \
    --use-cuda-graphs \
    --dataroot "D:\SAVE\files\Datasets\nuscenes-v1.0" \
    --version v1.0-trainval

# 结果保存在：[model]_results/openemma/[timestamp]/
# 记录输出目录名称，例如：20250102-040000
```

---

### **步骤 4：对比分析结果**

#### **A. 查看 ADE 指标**

```bash
# 版本 B（回退版）
type "[model]_results\openemma\20250102-030000\ade_results.jsonl"

# 版本 A（修复版）
type "[model]_results\openemma\20250102-040000\ade_results.jsonl"
```

#### **B. 对比示例**

```json
// 版本 B（回退版）- scene-0103
{
  "name": "scene-0103",
  "ade1s": 4.52,     // ← 预期：较高
  "ade2s": 5.23,
  "ade3s": 6.12,
  "avgade": 5.29,
  "error_1s": 8.45,
  "failure_rate_1s_frame": 0.15
}

// 版本 A（修复版）- scene-0103
{
  "name": "scene-0103",
  "ade1s": 2.26,     // ← 预期：降低 50%
  "ade2s": 3.67,     // ← 预期：降低 30%
  "ade3s": 5.20,     // ← 预期：降低 15%
  "avgade": 3.71,    // ← 预期：整体降低 30%
  "error_1s": 4.21,
  "failure_rate_1s_frame": 0.05
}
```

---

## 📊 预期对比结果

| 指标 | 版本 B（回退） | 版本 A（修复） | 改进幅度 |
|------|---------------|---------------|---------|
| **ADE-1s** | 高 | **↓50%** | 连续性修复主导 |
| **ADE-2s** | 中 | **↓30-40%** | 连续性修复 |
| **ADE-3s** | 相对低 | **↓15-25%** | 轻微改善 |
| **解析成功率** | 85% | **95%+** | 智能解析 |
| **Failure Rate** | 15% | **5-8%** | 综合改进 |

---

## 🔧 快速切换版本命令

### **切换到版本 B（回退版）**
```powershell
cd "D:\SAVE\files\Github\OpenEMMA"
# eval.py 和 main.py 当前已经是回退版本
```

### **切换到版本 A（修复版）**
```powershell
cd "D:\SAVE\files\Github\OpenEMMA"
Copy-Item Backup_WithFixes\eval.py eval.py -Force
Copy-Item Backup_WithFixes\main.py main.py -Force
Write-Host "已恢复修复版本"
```

---

## 📝 注意事项

1. **参数一致性**：两次实验必须使用完全相同的参数
   - 相同的 `--model-path`
   - 相同的 `--scenes` 或 `--split`
   - 相同的 `--quantization`
   - 相同的 `--use-cuda-graphs`

2. **场景选择建议**：
   - 快速测试：`--scenes scene-0103,scene-1077`（2个场景，约10分钟）
   - 完整评估：`--split val`（150个场景，约8-12小时）

3. **其他保持不变的优化**：
   - ✅ CUDA Graphs（加速但不影响 ADE）
   - ✅ 显存管理（不影响 ADE）
   - ✅ 代码架构（不影响 ADE）

4. **随机性控制**：
   - 如果模型有温度采样，结果可能有轻微波动
   - 建议运行多次取平均值

---

## 🎯 预期验证结论

如果对比结果符合预期，应该观察到：

✅ **版本 A（修复版）显著优于版本 B（回退版）**
- ADE-1s 降低 40-50%（最显著）
- ADE-2s 降低 25-35%
- ADE-3s 降低 10-20%
- Failure Rate 降低 50%+

这将**明确证明**连续性修复和解析优化的有效性！

---

## 📂 文件备份位置

```
OpenEMMA/
├── eval.py                      # 当前：版本 B（回退版）
├── main.py                      # 当前：版本 B（回退版）
└── Backup_WithFixes/           # 备份：版本 A（修复版）
    ├── eval.py                  # 包含所有修复
    └── main.py                  # 包含所有修复
```

---

**创建时间**：2025-12-26  
**目的**：量化预测连续性修复和解析优化对 ADE 的影响  
**状态**：✅ 已完成回退，可以开始对比实验

