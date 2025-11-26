"""
查看历史运行记录的工具脚本

使用方法：
    python view_runs.py                    # 查看所有运行记录
    python view_runs.py --latest 5         # 查看最近5次运行
    python view_runs.py --model llama      # 查看特定模型的运行
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict


def find_all_runs(base_dirs: List[str] = None) -> List[Dict]:
    """查找所有运行记录"""
    if base_dirs is None:
        # 默认搜索当前目录下的 *_results 文件夹
        base_dirs = [d for d in Path(".").glob("*_results") if d.is_dir()]
    
    runs = []
    for base_dir in base_dirs:
        for config_file in Path(base_dir).rglob("run_config.json"):
            metrics_file = config_file.parent / "runtime_metrics.json"
            
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                metrics = None
                if metrics_file.exists():
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                
                # 从路径提取时间戳
                timestamp_str = config_file.parent.name
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
                except:
                    timestamp = None
                
                runs.append({
                    "path": str(config_file.parent),
                    "timestamp": timestamp,
                    "config": config,
                    "metrics": metrics,
                })
            except Exception as e:
                print(f"[WARN] 无法读取 {config_file}: {e}")
    
    # 按时间排序
    runs.sort(key=lambda x: x["timestamp"] or datetime.min, reverse=True)
    return runs


def format_runtime(seconds: float) -> str:
    """格式化运行时间"""
    if seconds is None:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_memory(mb: float) -> str:
    """格式化显存"""
    if mb is None:
        return "N/A"
    if mb > 1024:
        return f"{mb/1024:.2f} GB"
    else:
        return f"{mb:.2f} MB"


def print_run_summary(run: Dict, index: int = None):
    """打印单次运行摘要"""
    config = run["config"]
    metrics = run["metrics"]
    
    print("=" * 80)
    if index is not None:
        print(f"运行 #{index}")
    if run["timestamp"]:
        print(f"时间: {run['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"路径: {run['path']}")
    print("-" * 80)
    
    # 配置信息
    print("配置:")
    print(f"  模型: {Path(config.get('model_path', 'N/A')).name}")
    print(f"  量化: {config.get('quantization', 'N/A')}")
    print(f"  方法: {config.get('method', 'N/A')}")
    print(f"  场景: {config.get('scenes', 'all') or 'all'}")
    print(f"  数据集: {config.get('version', 'N/A')}")
    
    # 性能指标
    if metrics:
        print("\n性能:")
        runtime = metrics.get("total_runtime_sec")
        gpu_mem = metrics.get("avg_gpu_memory_mb")
        samples = metrics.get("gpu_samples_count", 0)
        
        print(f"  运行时长: {format_runtime(runtime)}")
        print(f"  平均显存: {format_memory(gpu_mem)}")
        print(f"  采样次数: {samples}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="查看历史运行记录")
    parser.add_argument("--latest", type=int, help="只显示最近N次运行")
    parser.add_argument("--model", type=str, help="过滤特定模型（模糊匹配）")
    parser.add_argument("--method", type=str, help="过滤特定方法")
    parser.add_argument("--quantization", type=str, help="过滤特定量化方式")
    args = parser.parse_args()
    
    # 查找所有运行
    runs = find_all_runs()
    
    if not runs:
        print("未找到任何运行记录")
        return
    
    # 过滤
    if args.model:
        runs = [r for r in runs if args.model.lower() in r["config"].get("model_path", "").lower()]
    if args.method:
        runs = [r for r in runs if r["config"].get("method") == args.method]
    if args.quantization:
        runs = [r for r in runs if r["config"].get("quantization") == args.quantization]
    
    # 限制数量
    if args.latest:
        runs = runs[:args.latest]
    
    # 显示
    print(f"\n找到 {len(runs)} 条运行记录\n")
    for i, run in enumerate(runs, 1):
        print_run_summary(run, i)


if __name__ == "__main__":
    main()

