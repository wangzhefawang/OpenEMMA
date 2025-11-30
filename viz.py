"""可视化相关功能"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory_interpolation(
    ego_poses_world, ego_velocities, estimated_points, save_path
):
    """绘制轨迹插值结果"""
    plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], "r-", label="GT")
    plt.quiver(
        ego_poses_world[:, 0],
        ego_poses_world[:, 1],
        ego_velocities[:, 0],
        ego_velocities[:, 1],
        color="b",
    )
    plt.plot(estimated_points[:, 0], estimated_points[:, 1], "g-", label="Reconstruction")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_prediction_trajectory(fut_ego_traj_world, pred_traj, scene_name, frame_idx, ade, save_path):
    """绘制预测轨迹对比"""
    plt.plot(fut_ego_traj_world[:, 0], fut_ego_traj_world[:, 1], "r-", label="GT")
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], "b-", label="Pred")
    plt.legend()
    plt.title(f"Scene: {scene_name}, Frame: {frame_idx}, ADE: {ade}")
    plt.savefig(save_path)
    plt.close()


def save_frame_results(
    timestamp,
    scene_name,
    frame_idx,
    pred_traj,
    pred_curvatures,
    pred_speeds,
    scene_description,
    object_description,
    updated_intent,
    ade,
):
    """保存单帧的所有结果"""
    # Save the trajectory
    np.save(f"{timestamp}/{scene_name}_{frame_idx}_pred_traj.npy", pred_traj)
    np.save(f"{timestamp}/{scene_name}_{frame_idx}_pred_curvatures.npy", pred_curvatures)
    np.save(f"{timestamp}/{scene_name}_{frame_idx}_pred_speeds.npy", pred_speeds)

    # Save the descriptions
    with open(f"{timestamp}/{scene_name}_{frame_idx}_logs.txt", "w", encoding='utf-8') as f:
        f.write(f"Scene Description: {scene_description}\n")
        f.write(f"Object Description: {object_description}\n")
        f.write(f"Intent Description: {updated_intent}\n")
        f.write(f"Average Displacement Error: {ade}\n")


def save_scene_metrics(timestamp, scene_name, token, mean_ade1s, mean_ade2s, mean_ade3s):
    """保存场景级别的评估指标"""
    aveg_ade = np.mean([mean_ade1s, mean_ade2s, mean_ade3s])

    result = {
        "name": scene_name,
        "token": token,
        "ade1s": mean_ade1s,
        "ade2s": mean_ade2s,
        "ade3s": mean_ade3s,
        "avgade": aveg_ade,
    }

    with open(f"{timestamp}/ade_results.jsonl", "a", encoding='utf-8') as f:
        f.write(json.dumps(result))
        f.write("\n")

