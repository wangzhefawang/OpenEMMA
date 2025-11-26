"""配置文件 - 常量和参数解析"""
import argparse

# 时间窗口常量
OBS_LEN = 10  # 观测长度
FUT_LEN = 10  # 未来预测长度
TTL_LEN = OBS_LEN + FUT_LEN  # 总长度


def build_arg_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="OpenEMMA - 端到端多模态自动驾驶运动预测")
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=r"D:\SAVE\files\Models\Llama-3.2-11B-Vision-Instruct",
        help="VLM 模型路径或 HuggingFace 仓库名",
    )
    parser.add_argument(
        "--plot",
        type=bool,
        default=True,
        help="是否生成可视化结果",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default=r"D:\SAVE\files\Datasets\nuscenes-v1.0-mini",
        help="NuScenes 数据集根目录",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        help="NuScenes 数据集版本",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="openemma",
        help="使用的方法（openemma 或其他）",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="8bit",  # 与原始 main-251126.py 保持一致
        choices=["none", "4bit", "8bit"],
        help="选择 VLM 加载精度：none/4bit/8bit",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="逗号分隔的 scene 列表，如 scene-0103,scene-1077; 留空则跑全部",
    )
    
    return parser

