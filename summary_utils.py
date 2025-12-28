"""运行总结工具 - 生成完整的运行统计报告"""
import json
import os
from typing import Dict, List, Optional


def load_scene_results(ade_results_path: str) -> List[Dict]:
    """
    从 ade_results.jsonl 加载所有场景结果
    
    Args:
        ade_results_path: ade_results.jsonl 文件路径
        
    Returns:
        场景结果列表
    """
    results = []
    if not os.path.exists(ade_results_path):
        return results
    
    with open(ade_results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return results


def calculate_summary_metrics(scene_results: List[Dict]) -> Dict:
    """
    计算汇总指标
    
    Args:
        scene_results: 场景结果列表
        
    Returns:
        包含元数据、失败率指标和ADE指标的字典
    """
    if not scene_results:
        return {
            "metadata": {
                "total_scenes": 0,
                "failed_scenes": 0,
                "successful_scenes": 0
            },
            "failure_rate_metrics": {
                "scene_failure_rate": None,
                "mean_l2_1s": None,
                "mean_frame_failure_rate": None
            },
            "ade_metrics": {
                "mean_ade1s": None,
                "mean_ade2s": None,
                "mean_ade3s": None,
                "mean_avgade": None
            }
        }
    
    # 元数据统计
    total_scenes = len(scene_results)
    failed_scenes = sum(1 for r in scene_results if r.get("failure_1s_scene", 0) == 1)
    successful_scenes = total_scenes - failed_scenes
    
    # 收集各项指标（过滤掉 None 值）
    ade1s_values = [r["ade1s"] for r in scene_results if r.get("ade1s") is not None]
    ade2s_values = [r["ade2s"] for r in scene_results if r.get("ade2s") is not None]
    ade3s_values = [r["ade3s"] for r in scene_results if r.get("ade3s") is not None]
    avgade_values = [r["avgade"] for r in scene_results if r.get("avgade") is not None]
    
    error_1s_values = [r["error_1s"] for r in scene_results if r.get("error_1s") is not None]
    failure_rate_1s_frame_values = [
        r["failure_rate_1s_frame"] 
        for r in scene_results 
        if r.get("failure_rate_1s_frame") is not None
    ]
    
    # 计算平均值
    mean_ade1s = sum(ade1s_values) / len(ade1s_values) if ade1s_values else None
    mean_ade2s = sum(ade2s_values) / len(ade2s_values) if ade2s_values else None
    mean_ade3s = sum(ade3s_values) / len(ade3s_values) if ade3s_values else None
    mean_avgade = sum(avgade_values) / len(avgade_values) if avgade_values else None
    
    mean_l2_1s = sum(error_1s_values) / len(error_1s_values) if error_1s_values else None
    mean_frame_failure_rate = (
        sum(failure_rate_1s_frame_values) / len(failure_rate_1s_frame_values) 
        if failure_rate_1s_frame_values else None
    )
    
    # 场景级失败率（百分比）
    scene_failure_rate = (failed_scenes / total_scenes * 100) if total_scenes > 0 else None
    
    return {
        "metadata": {
            "total_scenes": total_scenes,
            "failed_scenes": failed_scenes,
            "successful_scenes": successful_scenes
        },
        "failure_rate_metrics": {
            "scene_failure_rate": round(scene_failure_rate, 2) if scene_failure_rate is not None else None,
            "mean_l2_1s": round(mean_l2_1s, 2) if mean_l2_1s is not None else None,
            "mean_frame_failure_rate": round(mean_frame_failure_rate * 100, 2) if mean_frame_failure_rate is not None else None
        },
        "ade_metrics": {
            "mean_ade1s": round(mean_ade1s, 2) if mean_ade1s is not None else None,
            "mean_ade2s": round(mean_ade2s, 2) if mean_ade2s is not None else None,
            "mean_ade3s": round(mean_ade3s, 2) if mean_ade3s is not None else None,
            "mean_avgade": round(mean_avgade, 2) if mean_avgade is not None else None
        }
    }


def format_runtime_info(metrics: Dict) -> Dict:
    """
    格式化运行时信息
    
    Args:
        metrics: 从 RuntimeMonitor 获取的原始指标
        
    Returns:
        格式化后的运行时信息
    """
    runtime_sec = metrics.get("total_runtime_sec")
    avg_gpu_mb = metrics.get("avg_gpu_memory_mb")
    
    # 格式化时间
    duration_formatted = None
    if runtime_sec is not None:
        hours = int(runtime_sec // 3600)
        minutes = int((runtime_sec % 3600) // 60)
        seconds = int(runtime_sec % 60)
        
        if hours > 0:
            duration_formatted = f"{hours}小时 {minutes}分钟 {seconds}秒"
        elif minutes > 0:
            duration_formatted = f"{minutes}分钟 {seconds}秒"
        else:
            duration_formatted = f"{seconds}秒"
    
    # GPU 显存信息
    gpu_info = {
        "enabled": avg_gpu_mb is not None,
        "avg_memory_mb": round(avg_gpu_mb, 2) if avg_gpu_mb is not None else None,
        "avg_memory_gb": round(avg_gpu_mb / 1024, 2) if avg_gpu_mb is not None else None,
        "samples": metrics.get("gpu_samples_count", 0)
    }
    
    return {
        "duration_seconds": round(runtime_sec, 2) if runtime_sec is not None else None,
        "duration_formatted": duration_formatted,
        "gpu_monitoring": gpu_info
    }


def generate_run_summary(
    timestamp_dir: str,
    run_config: Dict,
    runtime_metrics: Dict,
    cuda_stats: Optional[Dict] = None
) -> Dict:
    """
    生成完整的运行总结
    
    Args:
        timestamp_dir: 运行结果目录
        run_config: 运行配置（命令行参数）
        runtime_metrics: 运行时指标
        cuda_stats: CUDA Graphs 统计信息（可选）
        
    Returns:
        完整的运行总结字典
    """
    # 1. 加载场景结果
    ade_results_path = os.path.join(timestamp_dir, "ade_results.jsonl")
    scene_results = load_scene_results(ade_results_path)
    
    # 2. 计算汇总指标
    summary_metrics = calculate_summary_metrics(scene_results)
    
    # 3. 格式化运行时信息
    runtime_info = format_runtime_info(runtime_metrics)
    
    # 4. 组装最终结果
    summary = {
        "run_config": run_config,
        "results": summary_metrics,
        "runtime": runtime_info
    }
    
    # 5. 添加 CUDA Graphs 统计（如果有）
    if cuda_stats is not None:
        summary["cuda_graphs"] = {
            "enabled": True,
            "graph_hits": cuda_stats.get("graph_hits", 0),
            "graph_misses": cuda_stats.get("graph_misses", 0),
            "hit_rate": round(cuda_stats.get("hit_rate", 0), 4),
            "fallbacks": cuda_stats.get("fallbacks", 0),
            "total_captures": cuda_stats.get("total_captures", 0),
            "cached_graphs": cuda_stats.get("cached_graphs", 0),
            "max_graphs": 10  # 默认值，可以从配置中获取
        }
    else:
        summary["cuda_graphs"] = {
            "enabled": False
        }
    
    return summary


def save_run_summary(
    timestamp_dir: str,
    run_config: Dict,
    runtime_metrics: Dict,
    cuda_stats: Optional[Dict] = None,
    filename: str = "run_summary.json"
) -> str:
    """
    生成并保存运行总结到 JSON 文件
    
    Args:
        timestamp_dir: 运行结果目录
        run_config: 运行配置
        runtime_metrics: 运行时指标
        cuda_stats: CUDA Graphs 统计（可选）
        filename: 输出文件名（默认 run_summary.json）
        
    Returns:
        保存的文件路径
    """
    summary = generate_run_summary(timestamp_dir, run_config, runtime_metrics, cuda_stats)
    
    output_path = os.path.join(timestamp_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return output_path


def print_summary_report(summary: Dict):
    """
    打印格式化的总结报告
    
    Args:
        summary: 运行总结字典
    """
    print("\n" + "=" * 60)
    print("[SUMMARY] 运行总结报告")
    print("=" * 60)
    
    # 元数据
    metadata = summary["results"]["metadata"]
    print(f"\n[场景统计]")
    print(f"  总场景数: {metadata['total_scenes']}")
    print(f"  成功场景数: {metadata['successful_scenes']}")
    print(f"  失败场景数: {metadata['failed_scenes']}")
    
    # 失败率指标
    failure_metrics = summary["results"]["failure_rate_metrics"]
    print(f"\n[失败率指标]")
    print(f"  场景级失败率: {failure_metrics['scene_failure_rate']}%")
    print(f"  平均 1秒 L2 误差: {failure_metrics['mean_l2_1s']} m")
    print(f"  帧级平均失败率: {failure_metrics['mean_frame_failure_rate']}%")
    
    # ADE 指标
    ade_metrics = summary["results"]["ade_metrics"]
    print(f"\n[ADE 指标]")
    print(f"  平均 ADE (1s): {ade_metrics['mean_ade1s']} m")
    print(f"  平均 ADE (2s): {ade_metrics['mean_ade2s']} m")
    print(f"  平均 ADE (3s): {ade_metrics['mean_ade3s']} m")
    print(f"  平均 ADE (综合): {ade_metrics['mean_avgade']} m")
    
    # 运行时信息
    runtime = summary["runtime"]
    print(f"\n[运行时统计]")
    print(f"  总运行时长: {runtime['duration_formatted']} ({runtime['duration_seconds']}秒)")
    if runtime["gpu_monitoring"]["enabled"]:
        print(f"  GPU显存平均使用: {runtime['gpu_monitoring']['avg_memory_gb']} GB")
        print(f"  显存采样次数: {runtime['gpu_monitoring']['samples']}")
    
    # CUDA Graphs 统计
    cuda = summary.get("cuda_graphs", {})
    if cuda.get("enabled"):
        print(f"\n[CUDA Graphs 统计]")
        print(f"  缓存命中次数: {cuda['graph_hits']}")
        print(f"  缓存未命中次数: {cuda['graph_misses']}")
        print(f"  命中率: {cuda['hit_rate']:.2%}")
        print(f"  降级执行次数: {cuda['fallbacks']}")
        print(f"  总捕获图数: {cuda['total_captures']}")
        print(f"  当前缓存图数: {cuda['cached_graphs']}/{cuda['max_graphs']}")
    
    print("=" * 60 + "\n")

