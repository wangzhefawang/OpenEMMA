# CUDA Graphs 优化使用指南

## 📋 概述

CUDA Graphs 是一种 NVIDIA CUDA 技术，可以通过预先捕获计算图来减少 Python 调度开销，从而提升推理速度。在 OpenEMMA 中实现的 CUDA Graphs 优化预期可以带来 **10-40% 的速度提升**。

## 🚀 快速开始

### 启用 CUDA Graphs

在运行 OpenEMMA 时，只需添加 `--use-cuda-graphs` 参数：

```bash
python main.py \
    --model-path qwen \
    --dataroot [dir-of-nuscnse-dataset] \
    --version [version-of-nuscnse-dataset] \
    --method openemma \
    --use-cuda-graphs
```

### 调整预热次数

CUDA Graphs 需要预热以捕获计算图。默认预热次数为 3，可以通过 `--cuda-graphs-warmup` 调整：

```bash
python main.py \
    --model-path qwen \
    --dataroot [dir-of-nuscnse-dataset] \
    --version [version-of-nuscnse-dataset] \
    --method openemma \
    --use-cuda-graphs \
    --cuda-graphs-warmup 5
```

## 💡 工作原理

### 基本概念

CUDA Graphs 的核心思想是：
1. **捕获阶段**：首次执行时记录所有 CUDA 操作
2. **重放阶段**：后续执行直接重放记录的操作，跳过 Python 调度

### 实现特性

我们的实现包含以下智能特性：

#### 1. 多图缓存
- 自动为不同的输入形状缓存多个 graphs
- 最多缓存 10 个 graphs（LRU 策略淘汰）
- 相同输入形状自动重用已缓存的 graph

#### 2. 自动降级
- 遇到不支持的操作自动回退到普通推理
- 保证程序不会因为 CUDA Graphs 而崩溃
- 降级会记录警告日志

#### 3. 统计信息
运行结束后会输出详细的统计信息：
- 缓存命中次数
- 缓存未命中次数
- 命中率
- 降级执行次数
- 捕获的图数量

## 📊 性能对比

### 预期效果

| 场景 | 速度提升 | 显存增加 |
|------|---------|---------|
| 固定尺寸图像 + 相似文本长度 | 20-40% | 200-500MB |
| 动态文本长度 | 10-20% | 200-500MB |
| 首次运行（捕获阶段） | -5% ~ -10% | 200-500MB |

### 测试方法

建议进行 A/B 对比测试：

```bash
# 不使用 CUDA Graphs
python main.py \
    --model-path qwen \
    --dataroot [dataset-dir] \
    --version v1.0-mini \
    --split mini_val \
    --method openemma

# 使用 CUDA Graphs
python main.py \
    --model-path qwen \
    --dataroot [dataset-dir] \
    --version v1.0-mini \
    --split mini_val \
    --method openemma \
    --use-cuda-graphs
```

对比两次运行的 `runtime_metrics.json` 中的 `total_runtime_sec` 字段。

## ⚠️ 注意事项

### 适用场景

✅ **适合使用的情况：**
- 推理任务（非训练）
- 批量处理大量相似数据
- 图像尺寸固定
- 文本长度相对固定
- 显存充足（至少多 500MB）

❌ **不适合使用的情况：**
- 训练任务
- 单次或少量推理
- 输入形状变化很大
- 显存紧张
- 调试模式

### 限制和已知问题

1. **显存占用**
   - CUDA Graphs 会固定分配显存
   - 建议预留至少 500MB 额外显存
   - 与量化（4bit/8bit）一起使用时更容易 OOM

2. **动态输出**
   - VLM 生成任务的输出长度动态
   - CUDA Graphs 对动态输出的支持有限
   - 主要优化编码器部分，解码器优化效果有限

3. **首次运行开销**
   - 首次遇到新输入形状需要捕获 graph
   - 捕获过程会增加 1-2 秒延迟
   - 预热阶段会额外消耗 GPU 时间

4. **兼容性**
   - 需要 CUDA 11.0+
   - 需要 PyTorch 1.10+
   - 某些旧 GPU 可能不支持

## 📈 统计输出解读

运行结束后会看到类似输出：

```
============================================================
CUDA Graphs 统计信息
============================================================
缓存命中次数: 145
缓存未命中次数: 12
命中率: 92.36%
降级执行次数: 3
总捕获图数: 12
当前缓存图数: 10/10
============================================================
```

**指标说明：**
- **缓存命中次数**：重用已缓存 graph 的次数，越高越好
- **缓存未命中次数**：需要捕获新 graph 的次数
- **命中率**：命中次数 / 总调用次数，理想情况下应该 > 90%
- **降级执行次数**：无法使用 CUDA Graphs 的次数，应该尽量少
- **总捕获图数**：历史累计捕获的 graph 数量
- **当前缓存图数**：当前内存中缓存的 graph 数量

## 🔧 高级配置

### 修改最大缓存数量

如果发现缓存未命中率高，可以修改 `cuda_graphs_wrapper.py` 中的 `max_graphs` 参数：

```python
# 在 models.py 的 initialize_cuda_graphs 函数中
_cuda_graphs_wrapper = CUDAGraphsWrapper(
    max_graphs=20,  # 增加到 20
    warmup_iterations=warmup_iterations,
    enabled=use_cuda_graphs and torch.cuda.is_available()
)
```

注意：增加缓存数量会增加显存占用。

### 调整预热策略

如果遇到捕获失败，可以尝试增加预热次数：

```bash
python main.py \
    --use-cuda-graphs \
    --cuda-graphs-warmup 10  # 增加到 10 次
    ...
```

## 🐛 故障排查

### 常见问题

#### 1. OOM (Out of Memory)

**症状：** 运行时出现 CUDA out of memory 错误

**解决方案：**
- 减少 `max_graphs` 数量
- 使用更激进的量化（如 4bit）
- 增加 `torch.cuda.empty_cache()` 调用频率
- 关闭 CUDA Graphs

#### 2. 命中率低

**症状：** 统计显示命中率 < 50%

**可能原因：**
- 输入形状变化太大
- 文本长度差异很大
- max_graphs 设置太小

**解决方案：**
- 增加 `max_graphs`
- 预处理数据使输入更统一
- 考虑不使用 CUDA Graphs

#### 3. 速度没有提升

**症状：** 使用 CUDA Graphs 后速度反而下降

**可能原因：**
- 数据量太小，捕获开销占主导
- 输入差异太大，缓存命中率低
- GPU 利用率已经很高

**解决方案：**
- 只在批量处理时使用
- 检查命中率统计
- 进行性能分析

#### 4. 捕获失败

**症状：** 日志显示 "CUDA Graph 捕获失败"

**可能原因：**
- 动态控制流（if/while）
- CPU-GPU 同步操作
- 不支持的 CUDA 操作

**解决方案：**
- 检查模型代码
- 增加预热次数
- 使用简化版包装器 `SimpleCUDAGraphWrapper`

## 📚 技术参考

### 相关文档

- [CUDA Graphs 官方文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [PyTorch CUDA Graphs 支持](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)

### 实现细节

CUDA Graphs 包装器实现在 `cuda_graphs_wrapper.py` 中，核心类：

- **CUDAGraphsWrapper**：完整功能的包装器，支持多图缓存
- **SimpleCUDAGraphWrapper**：简化版，只缓存一个 graph

主要方法：
- `capture_and_replay()`：捕获或重放 graph
- `get_input_signature()`：生成输入签名用于缓存
- `get_statistics()`：获取统计信息

## 🤝 贡献

如果你有改进建议或发现 bug，欢迎提交 Issue 或 Pull Request！

## 📝 许可

本优化遵循 OpenEMMA 项目的 Apache 2.0 许可证。

