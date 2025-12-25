"""OpenEMMA 主程序入口 - 重构版本"""
import base64
import os
import re
import sys
from datetime import datetime
from math import atan2

import cv2
import numpy as np
import torch

from config import OBS_LEN, FUT_LEN, TTL_LEN, build_arg_parser
from models import load_vlm, prepare_image_payload, get_cuda_graphs_wrapper
from data_utils import load_nuscenes_dataset, parse_scene_filter, load_scene_data, get_split_scenes
from eval import GenerateMotion
from viz import (
    plot_trajectory_interpolation,
    plot_prediction_trajectory,
    save_frame_results,
    save_scene_metrics,
)
from utils import (
    EstimateCurvatureFromTrajectory,
    IntegrateCurvatureForPoints,
    OverlayTrajectory,
    WriteImageSequenceToVideo,
)
from runtime_monitor import RuntimeMonitor
from openemma.YOLO3D.inference import yolo3d_nuScenes


FAILURE_THRESHOLD_1S = 10.0  # 1 秒位置误差阈值（单位：米）
ONE_SECOND_INDEX = 1  # 在 dt≈0.5s 的设置下，1s 对应的 future step 下标（0-based）


def main():
    # 解析参数
    parser = build_arg_parser()
    args = parser.parse_args()

    # 初始化运行时监控（传入参数）
    monitor = RuntimeMonitor(run_args=vars(args))
    monitor.start()

    print(f"{args.model_path}")

    # 加载视觉语言模型
    model = None
    processor = None
    tokenizer = None
    try:
        model, tokenizer, processor = load_vlm(
            args.model_path, 
            quantization=args.quantization,
            use_cuda_graphs=args.use_cuda_graphs,
            warmup_iterations=args.cuda_graphs_warmup
        )
        assert (
            tokenizer is not None and model is not None
        ), "模型/分词器加载失败，请检查 --model-path"
        print(f"已从 {args.model_path} 加载模型。")
        
        if args.use_cuda_graphs:
            print(f"✨ CUDA Graphs 优化已启用（预热次数: {args.cuda_graphs_warmup}）")
        
        monitor.record_gpu_usage()
    except Exception as e:
        print("模型加载出现异常：", e)
        model = None
        processor = None
        tokenizer = None

    if model is None or tokenizer is None:
        print("致命错误：视觉语言模型未加载成功，程序终止。")
        sys.exit(1)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp = args.model_path + f"_results/{args.method}/" + timestamp
    os.makedirs(timestamp, exist_ok=True)
    
    # 保存命令行参数
    import json
    args_dict = vars(args)
    with open(os.path.join(timestamp, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=2)
    print(f"运行配置已保存到: {os.path.join(timestamp, 'run_config.json')}")

    # 加载数据集
    nusc = load_nuscenes_dataset(args.version, args.dataroot)

    # 解析场景过滤器
    scenes = nusc.scene
    
    # 获取 split 场景列表（如果指定）
    split_scene_names = get_split_scenes(args.split)
    
    # 获取手动指定的场景列表（如果有）
    allowed_scene_identifiers = parse_scene_filter(args.scenes, scenes)

    print(f"Number of scenes in dataset: {len(scenes)}")
    
    # 统计将要处理的场景数
    scenes_to_process = 0
    for scene in scenes:
        name = scene["name"]
        token = scene["token"]
        
        # 应用 split 过滤
        if split_scene_names and name not in split_scene_names:
            continue
            
        # 应用手动场景过滤
        if (
            allowed_scene_identifiers
            and name not in allowed_scene_identifiers
            and token not in allowed_scene_identifiers
        ):
            continue
            
        scenes_to_process += 1
    
    print(f"Number of scenes to process: {scenes_to_process}")

    # 遍历场景
    for scene in scenes:
        token = scene["token"]
        name = scene["name"]
        description = scene["description"]

        # 应用 split 过滤
        if split_scene_names and name not in split_scene_names:
            continue

        # 应用手动场景过滤
        if (
            allowed_scene_identifiers
            and name not in allowed_scene_identifiers
            and token not in allowed_scene_identifiers
        ):
            continue

        # 加载场景数据
        use_gpt = "gpt" in args.model_path
        front_camera_images, ego_poses, camera_params = load_scene_data(
            nusc, scene, use_gpt=use_gpt
        )

        scene_length = len(front_camera_images)
        print(f"\n\nScene {name} has {scene_length} frames")

        if scene_length < TTL_LEN:
            print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
            continue

        # 计算插值轨迹
        ego_poses_world = [ego_poses[t]["translation"][:3] for t in range(scene_length)]
        ego_poses_world = np.array(ego_poses_world)

        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
        ego_velocities[0] = ego_velocities[1]

        # 计算曲率
        ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
        ego_velocities_norm = np.linalg.norm(ego_velocities, axis=1)
        estimated_points = IntegrateCurvatureForPoints(
            ego_curvatures,
            ego_velocities_norm,
            ego_poses_world[0],
            atan2(ego_velocities[0][1], ego_velocities[0][0]),
            scene_length,
        )

        # 可视化插值结果
        if args.plot:
            plot_trajectory_interpolation(
                ego_poses_world,
                ego_velocities,
                estimated_points,
                f"{timestamp}/{name}_interpolation.jpg",
            )

        # 获取轨迹航点
        ego_traj_world = [ego_poses[t]["translation"][:3] for t in range(scene_length)]

        # 逐帧处理
        prev_intent = None
        cam_images_sequence = []
        ade1s_list = []
        ade2s_list = []
        ade3s_list = []
        error_1s_list = []  # 每一帧在 1 秒处的 L2 误差

        for i in range(scene_length - TTL_LEN):
            # 获取观测和未来数据
            obs_images = front_camera_images[i : i + OBS_LEN]
            obs_ego_poses = ego_poses[i : i + OBS_LEN]
            obs_camera_params = camera_params[i : i + OBS_LEN]
            obs_ego_traj_world = ego_traj_world[i : i + OBS_LEN]
            fut_ego_traj_world = ego_traj_world[i + OBS_LEN : i + TTL_LEN]
            obs_ego_velocities = ego_velocities[i : i + OBS_LEN]
            obs_ego_curvatures = ego_curvatures[i : i + OBS_LEN]

            # 获取车辆位置
            fut_start_world = obs_ego_traj_world[-1]
            curr_image = obs_images[-1]

            # 处理图像
            if use_gpt:
                img = cv2.imdecode(
                    np.frombuffer(base64.b64decode(curr_image), dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                img = yolo3d_nuScenes(img, calib=obs_camera_params[-1])[0]
            else:
                with open(os.path.join(curr_image), "rb") as image_file:
                    img = cv2.imdecode(
                        np.frombuffer(image_file.read(), dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )

            # 准备推理输入
            if use_gpt:
                obs_images_arg = front_camera_images[i : i + OBS_LEN]
            else:
                model_lower = args.model_path.lower()
                if any(key in model_lower for key in ("llava", "llama")):
                    obs_images_arg = prepare_image_payload(
                        curr_image, args=args, processor=processor, model=model
                    )
                else:
                    obs_images_arg = curr_image

            # 生成运动预测（最多重试3次）
            for rho in range(3):
                obs_images = obs_images_arg
                (
                    prediction,
                    scene_description,
                    object_description,
                    updated_intent,
                ) = GenerateMotion(
                    obs_images,
                    obs_ego_traj_world,
                    obs_ego_velocities,
                    obs_ego_curvatures,
                    prev_intent,
                    processor=processor,
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                )
                monitor.record_gpu_usage()
                
                # 定期清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 解析输出
                prev_intent = updated_intent
                
                # 调试：打印原始预测结果
                print(f"\n{'='*60}")
                print(f"原始 VLM 输出：")
                print(f"{prediction}")
                print(f"{'='*60}\n")
                
                # 智能解析：避免提取 prompt 中的观测数据
                pred_text = prediction.lower()  # 转小写进行匹配
                
                # 策略1：找到最后一次出现的 "future speeds and curvatures:" 标记
                # （因为 prompt 和 response 都可能包含这个短语）
                last_marker_pos = pred_text.rfind("future speeds and curvatures:")
                
                if last_marker_pos != -1:
                    # 从最后一个标记之后开始提取
                    pred_waypoints = prediction[last_marker_pos + len("future speeds and curvatures:"):]
                else:
                    # 策略2：尝试找到观测数据序列结束的位置
                    # 观测数据在 prompt 中的模式: "are [0.000,0.000], ..."
                    obs_marker_pos = pred_text.rfind("historical velocities and curvatures")
                    
                    if obs_marker_pos != -1:
                        # 跳过观测数据部分，寻找下一个句子的开始
                        pred_waypoints = prediction[obs_marker_pos:]
                        # 尝试移除到第一个完整句子
                        if "." in pred_waypoints:
                            # 跳过包含观测数据的句子
                            parts = pred_waypoints.split(".", 1)
                            if len(parts) > 1:
                                pred_waypoints = parts[1]
                    else:
                        # 策略3：如果都找不到，使用整个输出（向后兼容）
                        pred_waypoints = prediction
                
                # 提取所有数字对
                coordinates = re.findall(
                    r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints
                )
                
                print(f"[INFO] 从 VLM 输出中提取到 {len(coordinates)} 个坐标")
                
                if not coordinates == []:
                    break

            if coordinates == []:
                continue

            speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
            speed_curvature_pred = speed_curvature_pred[:10]
            # 格式化输出，与 Observed 格式一致
            pred_str = ", ".join([f"[{v:.3f},{k:.3f}]" for v, k in speed_curvature_pred])
            print(f"Got {len(speed_curvature_pred)} future actions: {pred_str}")

            # 预测轨迹
            pred_len = min(FUT_LEN, len(speed_curvature_pred))
            pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
            pred_speeds = np.array(speed_curvature_pred)[:, 0]
            pred_traj = np.zeros((pred_len, 3))
            pred_traj[:pred_len, :2] = IntegrateCurvatureForPoints(
                pred_curvatures,
                pred_speeds,
                fut_start_world,
                atan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0]),
                pred_len,
            )

            # 叠加轨迹到图像
            check_flag = OverlayTrajectory(
                img,
                pred_traj.tolist(),
                obs_camera_params[-1],
                obs_ego_poses[-1],
                color=(255, 0, 0),
                args=args,
            )

            # 计算 ADE
            fut_ego_traj_world = np.array(fut_ego_traj_world)
            ade = np.mean(np.linalg.norm(fut_ego_traj_world[:pred_len] - pred_traj, axis=1))

            pred1_len = min(pred_len, 2)
            ade1s = np.mean(
                np.linalg.norm(
                    fut_ego_traj_world[:pred1_len] - pred_traj[1 : pred1_len + 1], axis=1
                )
            )
            ade1s_list.append(ade1s)

            pred2_len = min(pred_len, 4)
            ade2s = np.mean(
                np.linalg.norm(fut_ego_traj_world[:pred2_len] - pred_traj[:pred2_len], axis=1)
            )
            ade2s_list.append(ade2s)

            pred3_len = min(pred_len, 6)
            ade3s = np.mean(
                np.linalg.norm(fut_ego_traj_world[:pred3_len] - pred_traj[:pred3_len], axis=1)
            )
            ade3s_list.append(ade3s)

            # 计算 1 秒处的单点 L2 误差（用于 Failure rate）
            if pred_len > ONE_SECOND_INDEX and fut_ego_traj_world.shape[0] > ONE_SECOND_INDEX:
                d_1s = np.linalg.norm(
                    pred_traj[ONE_SECOND_INDEX, :2] - fut_ego_traj_world[ONE_SECOND_INDEX, :2]
                )
                error_1s_list.append(float(d_1s))

            # 保存结果
            if args.plot == True:
                cam_images_sequence.append(img.copy())
                cv2.imwrite(f"{timestamp}/{name}_{i}_front_cam.jpg", img)

                # 绘制轨迹对比
                plot_prediction_trajectory(
                    fut_ego_traj_world,
                    pred_traj,
                    name,
                    i,
                    ade,
                    f"{timestamp}/{name}_{i}_traj.jpg",
                )

                # 保存所有结果
                save_frame_results(
                    timestamp,
                    name,
                    i,
                    pred_traj,
                    pred_curvatures,
                    pred_speeds,
                    scene_description,
                    object_description,
                    updated_intent,
                    ade,
                )

        # 保存场景级别指标
        mean_ade1s = float(np.mean(ade1s_list)) if ade1s_list else None
        mean_ade2s = float(np.mean(ade2s_list)) if ade2s_list else None
        mean_ade3s = float(np.mean(ade3s_list)) if ade3s_list else None

        if error_1s_list:
            mean_error_1s = float(np.mean(error_1s_list))
            failure_rate_1s_frame = float((np.array(error_1s_list) > FAILURE_THRESHOLD_1S).mean())
        else:
            mean_error_1s = None
            failure_rate_1s_frame = None

        # 场景级 Failure 标记：场景内 1 秒误差的平均值是否超过阈值
        failure_flag_1s_scene = (
            1 if (mean_error_1s is not None and mean_error_1s > FAILURE_THRESHOLD_1S) else 0
        )

        save_scene_metrics(
            timestamp,
            name,
            token,
            mean_ade1s,
            mean_ade2s,
            mean_ade3s,
            mean_error_1s,
            failure_rate_1s_frame,
            failure_flag_1s_scene,
        )

        # 生成视频
        if args.plot:
            WriteImageSequenceToVideo(cam_images_sequence, f"{timestamp}/{name}")

    # 输出运行时统计
    metrics = monitor.finish()
    metrics_path = os.path.join(timestamp, "runtime_metrics.json")
    monitor.dump(metrics_path)
    
    # 输出 CUDA Graphs 统计（如果启用）
    if args.use_cuda_graphs:
        cuda_wrapper = get_cuda_graphs_wrapper()
        if cuda_wrapper is not None:
            cuda_wrapper.print_statistics()
            
            # 将统计信息添加到 metrics
            cuda_stats = cuda_wrapper.get_statistics()
            metrics["cuda_graphs"] = cuda_stats
            monitor.dump(metrics_path)  # 重新保存包含 CUDA Graphs 统计的指标
    
    # 格式化输出
    print("\n" + "=" * 60)
    print("运行统计")
    print("=" * 60)
    
    if metrics["total_runtime_sec"] is not None:
        runtime_sec = metrics["total_runtime_sec"]
        hours = int(runtime_sec // 3600)
        minutes = int((runtime_sec % 3600) // 60)
        seconds = int(runtime_sec % 60)
        
        if hours > 0:
            print(f"总运行时长: {hours}小时 {minutes}分钟 {seconds}秒 ({runtime_sec:.2f}秒)")
        elif minutes > 0:
            print(f"总运行时长: {minutes}分钟 {seconds}秒 ({runtime_sec:.2f}秒)")
        else:
            print(f"总运行时长: {seconds}秒 ({runtime_sec:.2f}秒)")
    else:
        print("总运行时长: 未能计算")
    
    if metrics["avg_gpu_memory_mb"] is not None:
        gpu_mb = metrics["avg_gpu_memory_mb"]
        if gpu_mb >= 1024:
            gpu_gb = gpu_mb / 1024
            print(f"GPU显存平均使用: {gpu_gb:.2f} GB ({gpu_mb:.2f} MB)")
        else:
            print(f"GPU显存平均使用: {gpu_mb:.2f} MB")
        
        # 显示采样次数
        if "gpu_samples_count" in metrics:
            print(f"显存采样次数: {metrics['gpu_samples_count']}")
    else:
        print("GPU显存平均使用: GPU不可用")
    
    print(f"\n结果已保存到: {timestamp}")
    print(f"指标文件: {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

