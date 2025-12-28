# run_summary.json 完整运行记录文档

## 📋 概述

`run_summary.json` 文件保存了实验的完整运行记录，包括配置、结果和性能指标。该文件在每次运行结束时自动生成。

## 📁 文件位置

```
D:\SAVE\files\Models\Qwen2.5-VL-7B-Instruct_results/openemma/20251226-HHMMSS/
├── run_config.json         # 仅包含命令行参数（旧格式，保留兼容）
├── run_summary.json        # 完整运行总结（新格式，推荐使用）⭐
├── runtime_metrics.json    # 运行时指标（旧格式，保留兼容）
├── ade_results.jsonl       # 逐场景结果（每行一个场景）
└── ...其他文件
```

## 🗂️ 数据结构

### 完整结构示例

```json
{
  "run_config": {
    "model_path": "D:\\SAVE\\files\\Models\\Qwen2.5-VL-7B-Instruct",
    "plot": true,
    "dataroot": "D:\\SAVE\\files\\Datasets\\nuscenes-v1.0",
    "version": "v1.0-trainval",
    "method": "openemma",
    "quantization": "none",
    "split": "val",
    "scenes": "",
    "use_cuda_graphs": true,
    "cuda_graphs_warmup": 3,
    "max_new_tokens": null
  },
  "results": {
    "metadata": {
      "total_scenes": 150,
      "failed_scenes": 23,
      "successful_scenes": 127
    },
    "failure_rate_metrics": {
      "scene_failure_rate": 15.33,
      "mean_l2_1s": 8.67,
      "mean_frame_failure_rate": 12.45
    },
    "ade_metrics": {
      "mean_ade1s": 2.84,
      "mean_ade2s": 3.76,
      "mean_ade3s": 4.52,
      "mean_avgade": 3.71
    }
  },
  "runtime": {
    "duration_seconds": 8130.45,
    "duration_formatted": "2小时 15分钟 30秒",
    "gpu_monitoring": {
      "enabled": true,
      "avg_memory_mb": 18669.12,
      "avg_memory_gb": 18.23,
      "samples": 450
    }
  },
  "cuda_graphs": {
    "enabled": true,
    "graph_hits": 147,
    "graph_misses": 3,
    "hit_rate": 0.98,
    "fallbacks": 0,
    "total_captures": 3,
    "cached_graphs": 3,
    "max_graphs": 10
  }
}
```

## 📊 字段说明

### 1️⃣ run_config（运行配置）

保存所有命令行参数，用于实验重现。

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_path` | string | VLM 模型路径 |
| `plot` | boolean | 是否生成可视化 |
| `dataroot` | string | 数据集根目录 |
| `version` | string | 数据集版本 |
| `method` | string | 使用的方法（openemma/baseline） |
| `quantization` | string | 量化选项（none/4bit/8bit） |
| `split` | string | 数据集划分（train/val/test） |
| `scenes` | string | 指定场景列表（逗号分隔） |
| `use_cuda_graphs` | boolean | 是否启用 CUDA Graphs |
| `cuda_graphs_warmup` | integer | CUDA Graphs 预热次数 |
| `max_new_tokens` | integer/null | VLM 最大生成 token 数 |

### 2️⃣ results（评估结果）

#### metadata（元数据）

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_scenes` | integer | 总场景数 |
| `failed_scenes` | integer | 失败场景数（1秒误差 > 10m） |
| `successful_scenes` | integer | 成功场景数 |

#### failure_rate_metrics（失败率指标）

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `scene_failure_rate` | float | % | 场景级失败率 |
| `mean_l2_1s` | float | m | 平均 1 秒 L2 误差 |
| `mean_frame_failure_rate` | float | % | 帧级平均失败率 |

**计算说明：**
- `scene_failure_rate = (failed_scenes / total_scenes) * 100`
- `mean_l2_1s`：所有场景在 1 秒时刻的 L2 误差的平均值
- `mean_frame_failure_rate`：所有场景的帧级失败率的平均值

#### ade_metrics（ADE 指标）

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `mean_ade1s` | float | m | 1 秒平均位移误差 |
| `mean_ade2s` | float | m | 2 秒平均位移误差 |
| `mean_ade3s` | float | m | 3 秒平均位移误差 |
| `mean_avgade` | float | m | 综合平均位移误差 |

**计算说明：**
- ADE (Average Displacement Error)：预测轨迹与真实轨迹之间的平均欧氏距离
- `mean_avgade = mean(mean_ade1s, mean_ade2s, mean_ade3s)`

### 3️⃣ runtime（运行时统计）

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `duration_seconds` | float | 秒 | 总运行时长（秒） |
| `duration_formatted` | string | - | 格式化时间字符串 |

#### gpu_monitoring（GPU 监控）

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `enabled` | boolean | - | 是否启用 GPU 监控 |
| `avg_memory_mb` | float | MB | 平均显存使用（MB） |
| `avg_memory_gb` | float | GB | 平均显存使用（GB） |
| `samples` | integer | - | 显存采样次数 |

### 4️⃣ cuda_graphs（CUDA Graphs 统计）⭐

仅在启用 `--use-cuda-graphs` 时才有数据。

| 字段 | 类型 | 说明 |
|------|------|------|
| `enabled` | boolean | 是否启用 CUDA Graphs |
| `graph_hits` | integer | 缓存命中次数 |
| `graph_misses` | integer | 缓存未命中次数 |
| `hit_rate` | float | 命中率（0.0-1.0） |
| `fallbacks` | integer | 降级执行次数 |
| `total_captures` | integer | 总捕获图数 |
| `cached_graphs` | integer | 当前缓存图数 |
| `max_graphs` | integer | 最大缓存图数 |

**性能提示：**
- 命中率 > 90%：性能优化效果显著
- 命中率 50-90%：有一定提升
- 命中率 < 50%：建议禁用 CUDA Graphs

## 🔧 使用方法

### 运行程序

程序会自动生成 `run_summary.json`：

```bash
python main.py \
    --model-path D:\SAVE\files\Models\Qwen2.5-VL-7B-Instruct \
    --dataroot D:\SAVE\files\Datasets\nuscenes-v1.0 \
    --version v1.0-trainval \
    --method openemma \
    --split val \
    --use-cuda-graphs
```

运行结束后会看到：

```
============================================================
[SUMMARY] 运行总结报告
============================================================

[场景统计]
  总场景数: 150
  成功场景数: 127
  失败场景数: 23

[失败率指标]
  场景级失败率: 15.33%
  平均 1秒 L2 误差: 8.67 m
  帧级平均失败率: 12.45%

[ADE 指标]
  平均 ADE (1s): 2.84 m
  平均 ADE (2s): 3.76 m
  平均 ADE (3s): 4.52 m
  平均 ADE (综合): 3.71 m

[运行时统计]
  总运行时长: 2小时 15分钟 30秒 (8130.45秒)
  GPU显存平均使用: 18.23 GB
  显存采样次数: 450

[CUDA Graphs 统计]
  缓存命中次数: 147
  缓存未命中次数: 3
  命中率: 98.00%
  降级执行次数: 0
  总捕获图数: 3
  当前缓存图数: 3/10
============================================================

✅ 完整运行总结已保存到: [路径]/run_summary.json
```

### 读取和分析

使用 Python 读取：

```python
import json

# 读取运行总结
with open("run_summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)

# 访问数据
print(f"场景级失败率: {summary['results']['failure_rate_metrics']['scene_failure_rate']}%")
print(f"平均 ADE: {summary['results']['ade_metrics']['mean_avgade']} m")
print(f"CUDA Graphs 命中率: {summary['cuda_graphs']['hit_rate']:.2%}")
```

使用 `view_runs.py` 查看历史记录：

```bash
# 查看所有运行记录
python view_runs.py

# 查看最近 5 次运行
python view_runs.py --latest 5

# 查看特定模型的运行
python view_runs.py --model qwen
```

## 📈 用途

### 1. 可重现性
- 记录了完整的运行配置
- 可以准确复现实验结果

### 2. 结果对比
- 方便比较不同方法/配置的性能
- 支持 A/B 测试

### 3. 性能分析
- GPU 显存和运行时间统计
- CUDA Graphs 优化效果评估

### 4. 论文数据
- 直接提取评估指标
- 用于表格和图表制作

## 🆚 与旧格式的区别

| 项目 | 旧格式 | 新格式 |
|------|--------|--------|
| **配置保存** | `run_config.json` | ✅ 包含在 `run_summary.json` |
| **运行时指标** | `runtime_metrics.json` | ✅ 包含在 `run_summary.json` |
| **场景结果** | `ade_results.jsonl`（逐行） | ✅ 汇总在 `run_summary.json` |
| **元数据统计** | ❌ 无 | ✅ 有（total_scenes, failed_scenes） |
| **失败率指标** | ❌ 无汇总 | ✅ 有汇总 |
| **ADE 指标** | ❌ 无汇总 | ✅ 有汇总 |
| **CUDA Graphs** | 部分 | ✅ 完整统计 |
| **格式化时间** | ❌ 无 | ✅ 有 |

**推荐：** 优先使用 `run_summary.json`，它包含了所有信息且结构更清晰。

## 🛠️ 相关文件

| 文件 | 说明 |
|------|------|
| `summary_utils.py` | 核心实现：生成和保存运行总结 |
| `main.py` | 主程序：调用总结生成函数 |
| `test_summary.py` | 测试脚本：验证功能正确性 |
| `view_runs.py` | 查看工具：浏览历史运行记录 |

## 📝 示例：对比两次运行

```python
import json

def compare_runs(path1, path2):
    with open(path1, "r", encoding="utf-8") as f:
        run1 = json.load(f)
    with open(path2, "r", encoding="utf-8") as f:
        run2 = json.load(f)
    
    print("运行对比：")
    print(f"\n方法: {run1['run_config']['method']} vs {run2['run_config']['method']}")
    
    ade1 = run1['results']['ade_metrics']['mean_avgade']
    ade2 = run2['results']['ade_metrics']['mean_avgade']
    print(f"平均 ADE: {ade1} m vs {ade2} m")
    
    failure1 = run1['results']['failure_rate_metrics']['scene_failure_rate']
    failure2 = run2['results']['failure_rate_metrics']['scene_failure_rate']
    print(f"失败率: {failure1}% vs {failure2}%")
    
    time1 = run1['runtime']['duration_seconds']
    time2 = run2['runtime']['duration_seconds']
    speedup = time1 / time2
    print(f"运行时间: {time1:.0f}s vs {time2:.0f}s (加速 {speedup:.2f}x)")

# 示例
compare_runs(
    "run1/run_summary.json",
    "run2/run_summary.json"
)
```

## ❓ 常见问题

### Q: 为什么还保留 `run_config.json` 和 `runtime_metrics.json`？
**A:** 为了向后兼容。旧的脚本可能依赖这些文件。新代码应优先使用 `run_summary.json`。

### Q: 如果程序崩溃，会生成 `run_summary.json` 吗？
**A:** 不会。只有程序正常结束才会生成。但 `ade_results.jsonl` 是逐场景追加的，可以保留部分结果。

### Q: 如何批量分析多次运行的结果？
**A:** 使用 `view_runs.py` 或编写自己的分析脚本遍历 `*_results/` 目录下的所有 `run_summary.json` 文件。

### Q: 数值为 `null` 是什么意思？
**A:** 表示该指标无法计算（例如没有场景数据）或未启用相关功能。

---

**✨ 享受更便捷的实验管理！**

