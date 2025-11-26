# 故障排除指南

## 问题：bitsandbytes 8bit 量化错误

### 错误信息
```
RuntimeError: view size is not compatible with input tensor's size and stride
```

### 原因
这是 bitsandbytes 库在某些 PyTorch 版本中的已知兼容性问题。

### 解决方案

#### 方案 1: 使用 none 量化（推荐，最稳定）
```bash
python main.py --quantization none
```

#### 方案 2: 使用 4bit 量化（通常更稳定）
```bash
python main.py --quantization 4bit
```

#### 方案 3: 升级 bitsandbytes（如果方案1/2不满足需求）
```bash
pip install --upgrade bitsandbytes
pip install --upgrade torch
```

#### 方案 4: 修补 bitsandbytes 代码
找到文件：`D:\CODE\anaconda\Lib\site-packages\bitsandbytes\backends\cuda\ops.py`

第 145 行，将：
```python
outlier_cols = torch.argwhere(outliers.any(dim=0)).view(-1)
```
改为：
```python
outlier_cols = torch.argwhere(outliers.any(dim=0)).reshape(-1)
```

### 当前配置
- 默认量化已改为 `none`，避免此问题
- 如需量化以节省显存，建议先尝试 `4bit`

## 其他常见问题

### 显存不足
```bash
# 使用 4bit 量化
python main.py --quantization 4bit

# 或只处理特定场景
python main.py --scenes scene-0103
```

### 模型路径错误
确保 `--model-path` 指向正确的模型目录或 HuggingFace 仓库名。

### NuScenes 数据集路径错误
确保 `--dataroot` 指向包含 `v1.0-mini` 或 `v1.0-trainval` 的目录。

