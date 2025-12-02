# 历史速度数据来源详解

## 📊 数据流程图

```
nuScenes 数据集
    ↓
【第1步】加载 ego_pose（车辆位姿）
    ↓ data_utils.py: load_scene_data()
ego_poses = [pose1, pose2, pose3, ...]  # 每个 pose 包含 translation, rotation
    ↓
【第2步】提取 3D 位置坐标
    ↓ main.py: 第154行
ego_poses_world = [[x1,y1,z1], [x2,y2,z2], ...]  # 世界坐标系中的位置
    ↓
【第3步】计算速度（位置差分）
    ↓ main.py: 第157-159行
ego_velocities = [[vx1,vy1,vz1], [vx2,vy2,vz2], ...]  # 相邻帧的位移
    ↓
【第4步】滑动窗口截取观测窗口
    ↓ main.py: 第198行
obs_ego_velocities = ego_velocities[i:i+10]  # 取10帧历史数据
    ↓
【第5步】传递给 GenerateMotion
    ↓ main.py: 第242行
GenerateMotion(..., obs_ego_velocities, ...)
    ↓
【第6步】计算速度模长并格式化
    ↓ eval.py: 第112-116行
obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)  # 计算向量长度
obs_speed_curvature_str = "[v,k], [v,k], ..."  # 格式化为字符串
    ↓
【第7步】插入 Prompt
    ↓ eval.py: 第127行
"The 5 second historical velocities and curvatures are {obs_speed_curvature_str}"
    ↓
发送给 VLM 模型
```

---

## 🔍 详细代码追踪

### 1️⃣ 数据源：nuScenes 数据集

**文件**: `data_utils.py` 第 144 行

```python
# 从 nuScenes 数据集获取车辆位姿
pose = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
ego_poses.append(pose)
```

**ego_pose 数据结构**:
```python
{
    'token': 'xxx',
    'translation': [x, y, z],      # 3D 位置（米）
    'rotation': [w, x, y, z],      # 四元数旋转
    'timestamp': 1532402927814384  # 时间戳（微秒）
}
```

---

### 2️⃣ 提取位置坐标

**文件**: `main.py` 第 154 行

```python
ego_poses_world = [ego_poses[t]["translation"][:3] for t in range(scene_length)]
ego_poses_world = np.array(ego_poses_world)
```

**结果**: 
```python
# shape: (N, 3) - N帧，每帧3个坐标(x, y, z)
[[1000.0, 500.0, 1.5],   # 帧1
 [1000.5, 500.1, 1.5],   # 帧2
 [1001.0, 500.2, 1.5],   # 帧3
 ...]
```

---

### 3️⃣ 计算速度（差分法）

**文件**: `main.py` 第 157-159 行

```python
ego_velocities = np.zeros_like(ego_poses_world)
ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]  # 相邻帧位置差
ego_velocities[0] = ego_velocities[1]  # 第一帧用第二帧的速度填充
```

**计算原理**:
```
速度 = 位移 / 时间
     = (当前帧位置 - 前一帧位置) / 时间间隔

nuScenes 采样率: 2 Hz (每秒2帧)
时间间隔: 0.5 秒/帧
```

**示例**:
```python
# 位置
frame_1: [1000.0, 500.0, 1.5]
frame_2: [1001.0, 500.0, 1.5]

# 速度 (米/帧，每帧0.5秒)
velocity_2 = [1.0, 0.0, 0.0]  # x方向移动1米，0.5秒内

# 实际速度 = 1.0米 / 0.5秒 = 2.0 m/s
```

**⚠️ 注意**: 代码中的速度单位是**米/帧**（每0.5秒），不是米/秒！

---

### 4️⃣ 截取观测窗口

**文件**: `main.py` 第 198 行

```python
obs_ego_velocities = ego_velocities[i : i + OBS_LEN]  # OBS_LEN=10
```

**含义**: 
- 取连续 10 帧的速度数据
- 时间跨度: 10帧 × 0.5秒/帧 = **5秒历史**
- 这就是 prompt 中 "5 second historical velocities" 的来源！

---

### 5️⃣ 计算速度模长（标量）

**文件**: `eval.py` 第 112-114 行

```python
obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
```

**计算**:
```python
# 3D 向量速度 → 标量速度（模长）
velocity_vector = [vx, vy, vz]
speed = sqrt(vx² + vy² + vz²)

# 例如
[1.0, 0.0, 0.0] → speed = 1.0
[0.6, 0.8, 0.0] → speed = 1.0
```

---

### 6️⃣ 格式化为字符串

**文件**: `eval.py` 第 113-116 行

```python
obs_speed_curvature_str = [
    f"[{v:.3f},{k:.3f}]" 
    for v, k in zip(obs_velocities_norm, obs_curvatures)
]
obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)
```

**输出示例**:
```
"[0.000,0.000], [0.040,0.000], [0.283,-0.029], [0.666,-0.026], ..."
```

---

### 7️⃣ 插入 Prompt

**文件**: `eval.py` 第 127 行

```python
prompt = f"""...
The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}.
..."""
```

---

## 📐 数学公式

### 速度计算
```
v(t) = [x(t) - x(t-1), y(t) - y(t-1), z(t) - z(t-1)]
speed(t) = ||v(t)|| = sqrt(vx² + vy² + vz²)
```

### 曲率计算
见 `utils.py: EstimateCurvatureFromTrajectory()`

---

## 🎯 关键参数

| 参数 | 值 | 说明 |
|-----|---|------|
| **OBS_LEN** | 10 | 观测窗口长度（帧数） |
| **采样率** | 2 Hz | nuScenes 数据集采样频率 |
| **时间间隔** | 0.5 秒 | 相邻帧时间差 |
| **观测时长** | 5 秒 | 10帧 × 0.5秒 = 5秒 |
| **预测时长** | 5 秒 | FUT_LEN=10帧 × 0.5秒 = 5秒 |

---

## ⚠️ 重要说明

### 1. 速度单位不是米/秒！
代码中的 `ego_velocities` 单位是**米/帧**（每0.5秒）
- 如果要转换为米/秒，需要除以 0.5
- 但 prompt 中直接使用原始值

### 2. 第一帧的速度
```python
ego_velocities[0] = ego_velocities[1]
```
第一帧无法计算速度（没有前一帧），所以**复制第二帧的速度**

### 3. 车辆静止的情况
如果车辆在原地不动（如红灯等待）：
```python
位置不变 → ego_velocities = [0, 0, 0] → speed = 0.000
```

---

## 🔬 验证方法

想验证速度计算是否正确？在 `main.py` 第 159 行后添加：

```python
# 调试：打印前几帧的速度计算
print("=" * 60)
print("速度计算验证:")
for t in range(min(5, scene_length-1)):
    pos_curr = ego_poses_world[t]
    pos_next = ego_poses_world[t+1]
    vel_calc = pos_next - pos_curr
    vel_stored = ego_velocities[t+1]
    print(f"帧 {t}->{t+1}:")
    print(f"  位置差: {vel_calc}")
    print(f"  存储速度: {vel_stored}")
    print(f"  速度模长: {np.linalg.norm(vel_stored):.3f} m/0.5s")
print("=" * 60)
```

---

## 📚 总结

**历史速度数据来源**:
1. ✅ **真实数据**: 来自 nuScenes 数据集的 GPS/IMU 记录
2. ✅ **计算方法**: 相邻帧位置差分（简单但有效）
3. ✅ **时间窗口**: 滑动窗口取最近 5 秒（10帧）
4. ✅ **数据质量**: nuScenes 是高质量标注数据集，精度高

**不是模拟或合成的数据，而是真实自动驾驶车辆采集的轨迹！** 🚗

