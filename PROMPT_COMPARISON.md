# Prompt 对比分析

## 文件对比
- **main-original.py**: 原始版本
- **main-251126.py**: 2024年11月26日版本
- **当前版本 (eval.py)**: 最新重构版本

---

## 1️⃣ GenerateMotion - 运动预测 Prompt

### 🔴 原始版本 (main-original.py)

**OpenEMMA 方法:**
```python
prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
The scene is described as follows: {scene_description}. 
The identified critical objects are {object_description}. 
The car's intent is {intent_description}. 
The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
```

**非 OpenEMMA 方法:**
```python
prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
```

---

### 🟡 2024-11-26 版本 (main-251126.py)

**完全相同！** 与原始版本无差异。

---

### 🟢 当前版本 (eval.py) - **已改进**

**OpenEMMA 方法:**
```python
prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
The scene is described as follows: {scene_description}. 
The identified critical objects are {object_description}. 
The car's intent is {intent_description}. 
The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
Based on the historical data and scene context, predict the NEXT 10 future timesteps (5 seconds ahead). Only output the future predictions, do not repeat the historical data. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
```

**非 OpenEMMA 方法:**
```python
prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
Based on the historical data, predict the NEXT 10 future timesteps (5 seconds ahead). Only output the future predictions, do not repeat the historical data. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
```

**✨ 改进点:**
1. ✅ 添加 `"Based on the historical data and scene context, predict the NEXT 10 future timesteps (5 seconds ahead)."`
   - 明确预测的是"未来"而不是历史
   - 强调预测 10 个时间步
   
2. ✅ 添加 `"Only output the future predictions, do not repeat the historical data."`
   - **关键改进**: 防止模型回显历史观测数据
   - 解决了预测=观测的问题

---

## 2️⃣ SceneDescription Prompt

### 所有版本对比

| 版本 | Prompt 内容 | 差异 |
|-----|-----------|-----|
| **Original** | `"You are a autonomous driving labeller. You have access to these front-view camera images..."` | ❌ 语法错误：**a autonomous** |
| **251126** | **完全相同** | ❌ 语法错误未修复 |
| **当前版本** | **完全相同** | ❌ 语法错误未修复 |

**LLaVA 专用版本:**
- 所有版本都有特殊的 LLaVA prompt，要求 `"Provide a concise description"`
- 语法错误相同：`"You are an autonomous driving labeller"` ✅ (这里对了)

---

## 3️⃣ DescribeObjects Prompt

### 所有版本完全相同

```python
prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you."""
```

**问题:**
- ❌ `"a autonomous"` → 应该是 `"an autonomous"`
- ❌ `"a front-view camera images"` → 应该是 `"front-view camera images"`
- ❌ `"description of the that road user"` → 应该是 `"description of that road user"`

---

## 4️⃣ DescribeOrUpdateIntent Prompt

### 所有版本完全相同

**初始 Intent:**
```python
prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, describe the desired intent of the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""
```

**更新 Intent:**
```python
prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Explain your current intent: """
```

**LLaVA 版本:** 添加 `"Provide a concise description explanation"`

---

## 📊 总结

### 关键改进（仅在当前版本）

| 改进项 | 原因 | 影响 |
|-------|------|------|
| ✅ 添加 "predict the NEXT 10 future timesteps" | 明确预测目标 | ⭐⭐⭐⭐⭐ |
| ✅ 添加 "do not repeat the historical data" | **防止回显观测数据** | ⭐⭐⭐⭐⭐ |
| ✅ 添加 "Based on the historical data" | 强调推理基础 | ⭐⭐⭐ |

### 待修复的问题（所有版本都存在）

1. ❌ **语法错误**: `"a autonomous"` → `"an autonomous"`
2. ❌ **语法错误**: `"a front-view camera images"` → 单复数不一致
3. ❌ **语法错误**: `"the that road user"` → 重复冠词

### 各版本特点

| 版本 | 特点 | 适用场景 |
|-----|------|---------|
| **Original** | 基础版本 | ❌ 存在预测回显问题 |
| **251126** | 与 Original 完全相同 | ❌ 存在预测回显问题 |
| **Current** | **修复了预测回显** | ✅ **推荐使用** |

---

## 🎯 建议

### 立即采用
当前版本的 Prompt 改进是**必需的**，因为它解决了：
- 预测输出 = 观测数据的严重问题
- 模型不理解要预测"未来"的问题

### 可选优化
如果你追求完美，可以进一步修复语法错误（但对功能影响不大）：

```python
# 修复语法
"You are an autonomous driving labeller..."  # a → an
"You have access to front-view camera images..."  # 移除 a，复数一致
"description of that road user..."  # 移除 the
```

---

## 📝 修改历史

- **2024-11-26**: 创建 main-251126.py，但 Prompt 未改进
- **2024-12-02**: 重构为 eval.py，**添加防回显指令**
- **当前**: Prompt 已优化，解决预测=观测问题

