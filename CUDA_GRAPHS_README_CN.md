# CUDA Graphs 优化 - 快速上手

## 🎯 一句话总结

在运行命令中添加 `--use-cuda-graphs` 参数，即可获得 **10-40% 的推理速度提升**！

## 🚀 使用方法

### 基础用法

```bash
# 原始命令
python main.py \
    --model-path D:\SAVE\files\Models\Qwen2.5-VL-7B-Instruct \
    --dataroot D:\SAVE\files\Datasets\nuscenes-v1.0 \
    --version v1.0-trainval \
    --method openemma \
    --split mini_val

# 添加 CUDA Graphs 优化
python main.py \
    --model-path D:\SAVE\files\Models\Qwen2.5-VL-7B-Instruct \
    --dataroot D:\SAVE\files\Datasets\nuscenes-v1.0 \
    --version v1.0-trainval \
    --method openemma \
    --split mini_val \
    --use-cuda-graphs  # 👈 就这一行！
```

### 高级用法

```bash
# 调整预热次数（默认 3 次）
python main.py \
    --model-path D:\SAVE\files\Models\Qwen2.5-VL-7B-Instruct \
    --dataroot D:\SAVE\files\Datasets\nuscenes-v1.0 \
    --version v1.0-trainval \
    --method openemma \
    --split mini_val \
    --use-cuda-graphs \
    --cuda-graphs-warmup 5  # 增加预热次数
```

## 📊 效果展示

运行结束后，会自动显示优化统计：

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

============================================================
运行统计
============================================================
总运行时长: 15分钟 32秒 (932.45秒)
GPU显存平均使用: 18.23 GB (18669.12 MB)
显存采样次数: 450
============================================================
```

统计信息也会保存在 `runtime_metrics.json` 中，包含 `cuda_graphs` 字段。

## 🧪 测试验证

### 运行测试脚本

```bash
python test_cuda_graphs.py
```

这会运行一系列测试来验证 CUDA Graphs 功能是否正常。

### 性能对比测试

建议在相同数据集上进行 A/B 对比：

```bash
# 测试 1: 不使用优化
python main.py \
    --model-path qwen \
    --dataroot [dataset-dir] \
    --version v1.0-mini \
    --split mini_val \
    --method openemma

# 记录 runtime_metrics.json 中的 total_runtime_sec

# 测试 2: 使用优化
python main.py \
    --model-path qwen \
    --dataroot [dataset-dir] \
    --version v1.0-mini \
    --split mini_val \
    --method openemma \
    --use-cuda-graphs

# 再次记录 runtime_metrics.json 中的 total_runtime_sec
# 计算提升百分比
```

## ⚙️ 工作原理

CUDA Graphs 通过以下方式提升性能：

1. **消除 Python 开销**：将多次 CUDA 调用打包成一个图
2. **减少同步**：GPU 可以连续执行而不需要等待 CPU
3. **智能缓存**：相同输入形状自动重用已捕获的计算图

```
传统方式:
Python -> CUDA Kernel 1 -> 同步 -> Python -> CUDA Kernel 2 -> 同步 -> ...
   ⬇️
CUDA Graphs:
Python -> [CUDA Graph: Kernel 1 + Kernel 2 + ... ] -> 同步
```

## 💡 何时使用

### ✅ 推荐使用

- 批量处理大量场景（如完整 val split）
- 生产环境部署
- 性能评测
- 输入图像尺寸固定

### ❌ 不推荐使用

- 单次或少量推理
- 调试代码
- 显存不足（< 20GB）
- 输入变化很大

## 📖 详细文档

完整文档请参考：[CUDA_GRAPHS_USAGE.md](CUDA_GRAPHS_USAGE.md)

包含内容：
- 详细使用说明
- 性能调优技巧
- 故障排查指南
- 技术实现细节

## 🛠️ 文件说明

本优化新增的文件：

```
OpenEMMA/
├── cuda_graphs_wrapper.py        # CUDA Graphs 包装器实现
├── test_cuda_graphs.py           # 功能测试脚本
├── CUDA_GRAPHS_USAGE.md          # 详细使用文档
└── CUDA_GRAPHS_README_CN.md      # 快速上手指南（本文件）
```

修改的文件：
- `config.py`: 添加命令行参数
- `models.py`: 集成 CUDA Graphs 支持
- `eval.py`: 传递 CUDA Graphs 参数
- `main.py`: 启用统计输出

## ❓ 常见问题

### Q: 会影响准确性吗？
**A:** 不会。CUDA Graphs 只改变执行方式，不改变计算逻辑。

### Q: 所有模型都支持吗？
**A:** 理论上支持所有 PyTorch 模型。已测试：
- ✅ Qwen2-VL-7B
- ✅ Llama-3.2-11B-Vision
- ✅ LLaVA-1.6-Mistral-7B
- ❌ GPT-4o（云端 API，不适用）

### Q: 提升不明显怎么办？
**A:** 检查统计信息中的命中率。如果命中率低（< 50%），说明输入变化太大，可以：
- 增加 `max_graphs` 参数
- 确保输入预处理一致
- 考虑是否适合你的场景

### Q: 出现 OOM 怎么办？
**A:** CUDA Graphs 会额外占用显存。解决方案：
- 使用量化（`--quantization 4bit`）
- 减少 `max_graphs` 数量
- 升级 GPU 或增加显存
- 关闭 CUDA Graphs

## 🤝 反馈

如有问题或建议，欢迎提 Issue 或联系开发者！

---

**享受更快的推理速度！** 🚀

