"""测试运行总结功能"""
import json
import os
import tempfile
from summary_utils import (
    save_run_summary, 
    generate_run_summary,
    print_summary_report
)


def create_test_data(test_dir):
    """创建测试数据"""
    # 创建测试的 ade_results.jsonl
    ade_results_path = os.path.join(test_dir, "ade_results.jsonl")
    
    # 模拟 10 个场景的结果，其中 2 个失败
    test_scenes = [
        {
            "name": "scene-0001",
            "token": "token1",
            "ade1s": 2.5,
            "ade2s": 3.2,
            "ade3s": 4.1,
            "avgade": 3.27,
            "error_1s": 8.5,
            "failure_rate_1s_frame": 0.1,
            "failure_1s_scene": 0
        },
        {
            "name": "scene-0002",
            "token": "token2",
            "ade1s": 3.1,
            "ade2s": 4.0,
            "ade3s": 5.2,
            "avgade": 4.10,
            "error_1s": 12.3,  # 失败场景
            "failure_rate_1s_frame": 0.35,
            "failure_1s_scene": 1
        },
        {
            "name": "scene-0003",
            "token": "token3",
            "ade1s": 2.8,
            "ade2s": 3.5,
            "ade3s": 4.3,
            "avgade": 3.53,
            "error_1s": 7.8,
            "failure_rate_1s_frame": 0.08,
            "failure_1s_scene": 0
        },
        {
            "name": "scene-0004",
            "token": "token4",
            "ade1s": 2.2,
            "ade2s": 2.9,
            "ade3s": 3.6,
            "avgade": 2.90,
            "error_1s": 6.2,
            "failure_rate_1s_frame": 0.05,
            "failure_1s_scene": 0
        },
        {
            "name": "scene-0005",
            "token": "token5",
            "ade1s": 3.5,
            "ade2s": 4.3,
            "ade3s": 5.5,
            "avgade": 4.43,
            "error_1s": 15.2,  # 失败场景
            "failure_rate_1s_frame": 0.42,
            "failure_1s_scene": 1
        },
        {
            "name": "scene-0006",
            "token": "token6",
            "ade1s": 2.6,
            "ade2s": 3.3,
            "ade3s": 4.0,
            "avgade": 3.30,
            "error_1s": 7.5,
            "failure_rate_1s_frame": 0.07,
            "failure_1s_scene": 0
        },
        {
            "name": "scene-0007",
            "token": "token7",
            "ade1s": 2.9,
            "ade2s": 3.7,
            "ade3s": 4.6,
            "avgade": 3.73,
            "error_1s": 8.9,
            "failure_rate_1s_frame": 0.12,
            "failure_1s_scene": 0
        },
        {
            "name": "scene-0008",
            "token": "token8",
            "ade1s": 2.4,
            "ade2s": 3.1,
            "ade3s": 3.9,
            "avgade": 3.13,
            "error_1s": 7.1,
            "failure_rate_1s_frame": 0.06,
            "failure_1s_scene": 0
        },
        {
            "name": "scene-0009",
            "token": "token9",
            "ade1s": 2.7,
            "ade2s": 3.4,
            "ade3s": 4.2,
            "avgade": 3.43,
            "error_1s": 8.2,
            "failure_rate_1s_frame": 0.09,
            "failure_1s_scene": 0
        },
        {
            "name": "scene-0010",
            "token": "token10",
            "ade1s": 2.3,
            "ade2s": 3.0,
            "ade3s": 3.8,
            "avgade": 3.03,
            "error_1s": 6.8,
            "failure_rate_1s_frame": 0.04,
            "failure_1s_scene": 0
        }
    ]
    
    with open(ade_results_path, "w", encoding="utf-8") as f:
        for scene in test_scenes:
            f.write(json.dumps(scene) + "\n")
    
    # 模拟运行配置
    run_config = {
        "model_path": "D:\\SAVE\\files\\Models\\Qwen2.5-VL-7B-Instruct",
        "plot": True,
        "dataroot": "D:\\SAVE\\files\\Datasets\\nuscenes-v1.0",
        "version": "v1.0-trainval",
        "method": "openemma",
        "quantization": "none",
        "split": "mini_val",
        "scenes": "",
        "use_cuda_graphs": True,
        "cuda_graphs_warmup": 3,
        "max_new_tokens": None
    }
    
    # 模拟运行时指标
    runtime_metrics = {
        "total_runtime_sec": 932.45,
        "avg_gpu_memory_mb": 18669.12,
        "gpu_samples_count": 450
    }
    
    # 模拟 CUDA Graphs 统计
    cuda_stats = {
        "graph_hits": 145,
        "graph_misses": 12,
        "hit_rate": 0.9236,
        "fallbacks": 3,
        "total_captures": 12,
        "cached_graphs": 10
    }
    
    return run_config, runtime_metrics, cuda_stats


def test_summary_generation():
    """测试运行总结生成功能"""
    print("=" * 60)
    print("[TEST] 测试运行总结功能")
    print("=" * 60)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as test_dir:
        print(f"\n[OK] 创建测试目录: {test_dir}")
        
        # 创建测试数据
        run_config, runtime_metrics, cuda_stats = create_test_data(test_dir)
        print("[OK] 创建测试数据")
        
        # 生成运行总结
        summary_path = save_run_summary(
            timestamp_dir=test_dir,
            run_config=run_config,
            runtime_metrics=runtime_metrics,
            cuda_stats=cuda_stats
        )
        print(f"[OK] 生成运行总结: {summary_path}")
        
        # 读取并验证
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        # 打印总结报告
        print_summary_report(summary)
        
        # 验证数据结构
        print("\n[CHECK] 验证数据结构:")
        
        # 检查顶层字段
        required_fields = ["run_config", "results", "runtime", "cuda_graphs"]
        for field in required_fields:
            if field in summary:
                print(f"  [OK] {field}: 存在")
            else:
                print(f"  [FAIL] {field}: 缺失")
        
        # 检查 results 子字段
        print("\n  results 子字段:")
        results_fields = ["metadata", "failure_rate_metrics", "ade_metrics"]
        for field in results_fields:
            if field in summary.get("results", {}):
                print(f"    [OK] {field}: 存在")
            else:
                print(f"    [FAIL] {field}: 缺失")
        
        # 打印完整的 JSON 结构（格式化）
        print("\n" + "=" * 60)
        print("[JSON] 完整 JSON 结构预览:")
        print("=" * 60)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        
        # 验证关键数值
        print("\n" + "=" * 60)
        print("[VERIFY] 关键指标验证:")
        print("=" * 60)
        
        metadata = summary["results"]["metadata"]
        print(f"  总场景数: {metadata['total_scenes']} (预期: 10)")
        print(f"  失败场景数: {metadata['failed_scenes']} (预期: 2)")
        print(f"  成功场景数: {metadata['successful_scenes']} (预期: 8)")
        
        failure_metrics = summary["results"]["failure_rate_metrics"]
        print(f"  场景级失败率: {failure_metrics['scene_failure_rate']}% (预期: 20.0%)")
        
        # 验证正确性
        assert metadata['total_scenes'] == 10, "总场景数不正确"
        assert metadata['failed_scenes'] == 2, "失败场景数不正确"
        assert failure_metrics['scene_failure_rate'] == 20.0, "场景级失败率不正确"
        
        print("\n[OK] 所有测试通过！")


def test_summary_generation_with_empty_data():
    """测试空数据情况"""
    print("\n" + "=" * 60)
    print("[TEST] 测试空数据场景")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as test_dir:
        # 创建空的配置
        run_config = {"method": "test"}
        runtime_metrics = {
            "total_runtime_sec": 100.0,
            "avg_gpu_memory_mb": 5000.0,
            "gpu_samples_count": 10
        }
        
        # 不创建 ade_results.jsonl 文件
        summary = generate_run_summary(
            timestamp_dir=test_dir,
            run_config=run_config,
            runtime_metrics=runtime_metrics
        )
        
        print("[OK] 成功处理空数据场景")
        print(f"  总场景数: {summary['results']['metadata']['total_scenes']}")
        
        assert summary['results']['metadata']['total_scenes'] == 0
        assert summary['results']['ade_metrics']['mean_ade1s'] is None
        
        print("[OK] 空数据测试通过！")


if __name__ == "__main__":
    test_summary_generation()
    test_summary_generation_with_empty_data()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 所有测试完成！")
    print("=" * 60)

