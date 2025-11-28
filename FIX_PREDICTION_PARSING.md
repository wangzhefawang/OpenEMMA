# 修复预测输出解析问题

## 问题描述

在运行 OpenEMMA 时，发现**预测的未来动作与观测到的速度/曲率完全一样**。例如：

```
Observed Speed and Curvature: [0.000,0.000], [0.000,0.000], ..., [0.040,0.000], [0.283,-0.029], [0.666,-0.026]
Got 10 future actions: [[0.0, 0.0], [0.0, 0.0], ..., [0.04, 0.0], [0.283, -0.029], [0.666, -0.026]]
```

## 根本原因

这是由于以下三个问题造成的：

### 1. **VLM 输出包含完整 Prompt**
某些视觉语言模型（特别是 Qwen 和 Llama 系列）在生成输出时，会在响应中**回显输入 prompt 的内容**，包括：
- 观测到的历史速度和曲率数据
- Prompt 中的指示文本

### 2. **正则表达式匹配过于宽泛**
原始代码使用的正则表达式会匹配**所有**的 `[数字, 数字]` 格式：

```python
coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints)
```

这会同时提取：
- Prompt 中的观测数据（如果被回显）
- 模型生成的预测数据

### 3. **只取前 10 个导致错误**
代码只取前 10 个匹配的坐标：

```python
speed_curvature_pred = speed_curvature_pred[:10]
```

**如果观测序列有 10 个点，那么提取的就全是观测数据，而不是预测！**

## 解决方案

我们采用了三管齐下的方法：

### 📝 方案 1：改进输出解析逻辑

**文件**: `main.py` (第 256-280 行)

**改进内容**:
1. 添加调试输出，显示 VLM 的原始输出
2. 使用 `rfind()` 查找**最后一次**出现的 "future speeds and curvatures:" 标记
3. 实现多策略解析：
   - 策略1：从最后一个标记之后提取
   - 策略2：跳过 "historical velocities and curvatures" 部分
   - 策略3：向后兼容，使用整个输出

```python
# 找到最后一次出现的标记（因为 prompt 和 response 都可能包含）
last_marker_pos = pred_text.rfind("future speeds and curvatures:")

if last_marker_pos != -1:
    # 从最后一个标记之后开始提取
    pred_waypoints = prediction[last_marker_pos + len("future speeds and curvatures:"):]
```

### 🎯 方案 2：改进 Prompt

**文件**: `eval.py` (第 122-132 行)

**改进内容**:
在 prompt 中明确要求模型：
- 只输出**未来预测**
- 不要重复历史数据
- 强调预测的是"NEXT 10 future timesteps"

**修改前**:
```
Generate the predicted future speeds and curvatures...
```

**修改后**:
```
Based on the historical data and scene context, predict the NEXT 10 future timesteps (5 seconds ahead). 
Only output the future predictions, do not repeat the historical data.
```

### ⚙️ 方案 3：优化生成参数

**文件**: `models.py` (多处)

**改进内容**:
为 Qwen 和 Llama 模型添加采样参数，提高输出多样性和质量：

```python
generated_ids = model.generate(
    **inputs, 
    max_new_tokens=max_tokens,
    do_sample=True,        # 启用采样
    temperature=0.7,       # 控制随机性
    top_p=0.9,            # 核采样
)
```

**参数说明**:
- `do_sample=True`: 启用随机采样而非贪心解码
- `temperature=0.7`: 较高的值（0.7）提供更多样化的输出
- `top_p=0.9`: 从累积概率前 90% 的 token 中采样

## 验证方法

运行程序后，你会看到以下调试信息：

```
============================================================
原始 VLM 输出：
[这里会显示模型的完整输出]
============================================================

[INFO] 从 VLM 输出中提取到 X 个坐标
Got 10 future actions: [...]
```

**检查要点**:
1. 查看原始 VLM 输出，确认它是否包含 prompt 回显
2. 确认提取的坐标数量合理（通常应该是 10 个或更多）
3. 验证预测动作与观测数据不同（特别是当车辆在运动时）

## 预期效果

### 修复前
- 预测完全复制观测数据
- ADE (平均位移误差) 可能异常低（因为"预测"只是历史数据的延续）
- 对于静止启动场景，预测仍然是静止

### 修复后
- 预测基于场景理解和历史趋势
- 预测能够捕捉加速、减速、转向等运动变化
- 模型真正发挥 VLM 的推理能力

## 附加说明

### 关于静止场景
如果车辆**确实处于静止状态**（如红灯等待），那么预测为 `[0.0, 0.0]` 是合理的。问题只出现在：
- 车辆正在运动
- 但预测仍然完全复制观测数据

### 关于调试输出
如果你不想看到详细的调试信息，可以注释掉 `main.py` 中的这些行：

```python
# print(f"\n{'='*60}")
# print(f"原始 VLM 输出：")
# print(f"{prediction}")
# print(f"{'='*60}\n")
```

## 相关文件

- `main.py`: 主程序入口，包含输出解析逻辑
- `eval.py`: 评估和推理函数，包含 prompt 构建
- `models.py`: 模型加载和推理，包含生成参数设置

## 作者说明

此问题由用户在实际运行中发现，通过分析终端输出和代码逻辑定位根本原因，并实施了系统性的修复方案。

修复日期: 2025-11-28

