"""数据加载和预处理相关功能"""
import base64
import os
import re
from typing import List, Set, Tuple
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


def load_nuscenes_dataset(version: str, dataroot: str):
    """加载 NuScenes 数据集"""
    return NuScenes(version=version, dataroot=dataroot)


def get_split_scenes(split: str) -> Set[str]:
    """
    获取 NuScenes 官方 split 的场景名称集合
    
    Args:
        split: split 名称（train/val/trainval/test/mini_train/mini_val 等）
        
    Returns:
        该 split 包含的场景名称集合
    """
    if not split:
        return set()
    
    try:
        splits = create_splits_scenes()
        
        # 处理 trainval 特殊情况（train + val）
        if split == "trainval":
            if "train" in splits and "val" in splits:
                scene_names = set(splits["train"]) | set(splits["val"])
                print(f"已加载 NuScenes '{split}' split: {len(scene_names)} 个场景 (train={len(splits['train'])} + val={len(splits['val'])})")
            else:
                print(f"警告：无法找到 train 或 val split，将处理所有场景")
                return set()
        elif split in splits:
            scene_names = set(splits[split])
            print(f"已加载 NuScenes '{split}' split: {len(scene_names)} 个场景")
        else:
            available_splits = ", ".join(splits.keys())
            print(f"警告：未知的 split '{split}'，可用的 split: {available_splits}")
            return set()
            
        return scene_names
        
    except Exception as e:
        print(f"警告：加载 split 信息失败: {e}")
        return set()


def parse_scene_filter(scenes_arg: str, all_scenes: list) -> Set[str]:
    """
    解析 --scenes 参数，返回允许的 scene 标识符集合
    
    Args:
        scenes_arg: 逗号分隔的 scene 列表字符串
        all_scenes: 所有可用的 scene 列表
        
    Returns:
        允许的 scene 标识符集合
    """
    allowed_scene_identifiers = set()
    if not scenes_arg:
        return allowed_scene_identifiers
        
    normalized_scene_arg = scenes_arg.replace(",", ",").replace(";", ",")
    for item in re.split(r"[,\s]+", normalized_scene_arg):
        item = item.strip()
        if item:
            allowed_scene_identifiers.add(item)
            
    if allowed_scene_identifiers:
        matched_scene_names = []
        matched_identifiers = set()
        for scene in all_scenes:
            if (
                scene["name"] in allowed_scene_identifiers
                or scene["token"] in allowed_scene_identifiers
            ):
                matched_scene_names.append(scene["name"])
                matched_identifiers.add(scene["name"])
                matched_identifiers.add(scene["token"])
        missing_identifiers = allowed_scene_identifiers - matched_identifiers
        if matched_scene_names:
            print(
                f"按 --scenes 过滤后将处理 {len(matched_scene_names)} 个 scene：{', '.join(matched_scene_names)}"
            )
        else:
            print(
                f"警告：--scenes 参数未匹配到任何 scene，将默认处理全部 {len(all_scenes)} 个 scene。"
            )
        if missing_identifiers:
            print(f"警告：未找到以下 scene 标识：{', '.join(sorted(missing_identifiers))}")
            
    return allowed_scene_identifiers


def load_scene_data(
    nusc: NuScenes,
    scene: dict,
    use_gpt: bool = False,
) -> Tuple[List, List, List]:
    """
    加载单个 scene 的所有图像、位姿和相机参数
    
    Args:
        nusc: NuScenes 数据集对象
        scene: scene 字典
        use_gpt: 是否使用 GPT 模型（需要 base64 编码）
        
    Returns:
        (front_camera_images, ego_poses, camera_params) 元组
    """
    front_camera_images = []
    ego_poses = []
    camera_params = []
    
    first_sample_token = scene["first_sample_token"]
    last_sample_token = scene["last_sample_token"]
    curr_sample_token = first_sample_token
    
    while True:
        sample = nusc.get("sample", curr_sample_token)

        # Get the front camera image of the sample.
        cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])

        if use_gpt:
            with open(
                os.path.join(nusc.dataroot, cam_front_data["filename"]), "rb"
            ) as image_file:
                front_camera_images.append(
                    base64.b64encode(image_file.read()).decode("utf-8")
                )
        else:
            front_camera_images.append(
                os.path.join(nusc.dataroot, cam_front_data["filename"])
            )

        # Get the ego pose of the sample.
        pose = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
        ego_poses.append(pose)

        # Get the camera parameters of the sample.
        camera_params.append(
            nusc.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"])
        )

        # Advance the pointer.
        if curr_sample_token == last_sample_token:
            break
        curr_sample_token = sample["next"]
        
    return front_camera_images, ego_poses, camera_params

