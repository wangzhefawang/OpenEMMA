import base64
import requests
import time
import random
import io
import base64
from math import atan2
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from pyquaternion import Quaternion
from scipy.integrate import cumulative_trapezoid

random.seed(42)

KEY = "<your-api-key>"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def query_gpt4(question, api_key=None, image_path=None, proxy='openai', sys_message=None):

    if proxy == "ohmygpt":
        request_url = "https://aigptx.top/v1/chat/completions"
    elif proxy == "openai":
        request_url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": 'Bearer ' + api_key,
    }

    if image_path is not None:
        base64_image = encode_image(image_path)
        if sys_message is not None:
            params = {
                "messages": [
                    {
                    "role": "system", 
                    "content": sys_message
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "model": 'gpt-4o',
                "temperature": 0.0
            }
        else:

            params = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "model": 'gpt-4o-mini-2024-07-18',
                "temperature": 0.0
            }
    else:
        if sys_message is not None:
            params = {
                "messages": [

                    {
                        "role": "system", 
                        "content": sys_message
                    },
                    {
                        "role": 'user',
                        "content": question
                    }
                ],
                "model": 'gpt-4o',
                "temperature": 0.0
            }
        else:
            params = {
                "messages": [
                    {
                        "role": 'user',
                        "content": question
                    }
                ],
                "model": 'gpt-4o',
                "temperature": 0.0
            }


    received = False
    while not received:
        try:
            response = requests.post(
                request_url,
                headers=headers,
                json=params,
                stream=False
            )
            res = response.json()
            res_content = res['choices'][0]['message']['content']
            received = True
        except:
            time.sleep(1)
    return res_content


def PlotBase64Image(image: str):
    i = base64.b64decode(image)
    i = io.BytesIO(i)
    i = mpimg.imread(i, format='JPG')

    plt.imshow(i, interpolation='nearest')
    plt.show()



def TransformPoint(point, transform):
    """ Transform a 3D point using a transformation matrix. """
    if isinstance(point, list):
        point = np.array(point)

    if point.shape[-1] == 3:
        point = np.append(point, 1)
    transformed_point = transform @ point
    return transformed_point[:3]

def FormTransformationMatrix(translation, rotation):
    """ Create a transformation matrix from translation and rotation (as a quaternion). """
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T

def ProjectEgoToImage(points_3d: np.array, K):
    """ Project 3D points to 2D using camera intrinsic matrix K. """
    # Filter out points that are behind the camera
    points_3d = points_3d[points_3d[:, 2] > 0]

    # Project the remaining points
    points_2d = np.dot(K, points_3d.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2][:, np.newaxis]  # Normalize by depth
    return points_2d

def ProjectWorldToImage(points3d_world: list, cam_to_ego, ego_to_world):
    # Plot the waypoints.

    T_ego_global = FormTransformationMatrix(ego_to_world['translation'], Quaternion(ego_to_world['rotation']))
    T_cam_ego = FormTransformationMatrix(cam_to_ego['translation'], Quaternion(cam_to_ego['rotation']))
    T_cam_global = T_ego_global @ T_cam_ego
    T_global_cam = np.linalg.inv(T_cam_global)

    points3d_cam = [TransformPoint(point, T_global_cam) for point in points3d_world]

    points3d_img = ProjectEgoToImage(np.array(points3d_cam), cam_to_ego['camera_intrinsic'])

    return points3d_img


def OffsetTrajectory3D(points, offset_distance, eps=1e-6):
    """
    Offsets a 3D trajectory by a specified distance normal to the trajectory.

    Parameters:
        points (np.ndarray): n x 3 array representing the 3D trajectory (x, y, z).
        offset_distance (float): Distance to offset the trajectory.

    Returns:
        np.ndarray: Offset trajectory as an n x 3 array.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 2:
        # 太短的轨迹，没法算切线，直接原样返回
        return points.copy()

    # 1. 用差分近似切向量（比 gradient 稍微稳定一点）
    tangents = np.zeros_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]

    # 2. 归一化前先算范数
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    zero_mask = norms < eps  # 认为这是“静止/几乎没动”的点

    # 2.1 对于范数太小的点，尽量用前后非零切向来补
    for i in range(n):
        if zero_mask[i]:
            if i > 0 and not zero_mask[i - 1]:
                tangents[i] = tangents[i - 1]
            elif i < n - 1 and not zero_mask[i + 1]:
                tangents[i] = tangents[i + 1]

    # 2.2 再算一遍范数
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    zero_mask = norms < eps

    # 如果还是 0，就给一个默认方向，比如沿 x 轴
    tangents[zero_mask.squeeze(-1)] = np.array([1.0, 0.0, 0.0])
    norms[zero_mask] = 1.0

    # 2.3 真正归一化
    tangents = tangents / norms

    # 3. 计算法向量：先用 z 轴作为参考
    ref_z = np.array([0.0, 0.0, 1.0])
    normals = np.cross(tangents, ref_z)
    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)

    # 3.1 如果切向刚好平行 z 轴，法向会变成 0，换一个参考向量再来一次
    parallel_mask = normal_norms < eps
    if np.any(parallel_mask):
        ref_y = np.array([0.0, 1.0, 0.0])
        normals[parallel_mask.squeeze(-1)] = np.cross(
            tangents[parallel_mask.squeeze(-1)], ref_y
        )
        normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)

    # 3.2 再次防止 0，给默认值
    normal_norms[normal_norms < eps] = 1.0
    normals = normals / normal_norms

    # 4. 计算偏移轨迹
    offset_points = points + offset_distance * normals
    return offset_points

def OverlayTrajectory(img, points3d_world: list, cam_to_ego, ego_to_world, color=(0, 0, 255), args=None):

    # Construct left/right boundaries.
    points3d_left_world = OffsetTrajectory3D(np.array(points3d_world), -1.73 / 2)
    points3d_right_world = OffsetTrajectory3D(np.array(points3d_world), 1.73 / 2)

    # Project the waypoints to the image.
    points3d_img = ProjectWorldToImage(points3d_world, cam_to_ego, ego_to_world)
    points3d_left_img = ProjectWorldToImage(points3d_left_world.tolist(), cam_to_ego, ego_to_world)
    points3d_right_img = ProjectWorldToImage(points3d_right_world.tolist(), cam_to_ego, ego_to_world)

    if args.plot:
        # Overlay the waypoints on the image.
        for i in range(len(points3d_img) - 1):
            cv2.circle(img, tuple(points3d_img[i].astype(int)), radius=6, color=color, thickness=-1)

        # # Draw lines.
        # for i in range(len(points3d_img) - 1):
        #     cv2.line(img, tuple(points3d_img[i].astype(int)), tuple(points3d_img[i+1].astype(int)), color, 2)

    # Draw sweep area polygon between the boundaries.
    frame = np.zeros_like(img)
    polygon = np.vstack((np.array(points3d_left_img), np.array(points3d_right_img)[::-1])).astype(np.int32)
    check_flag = False
    if polygon.size == 0:
        check_flag = True
        return check_flag
    if args.plot:
        cv2.fillPoly(frame, [polygon], color=color)  # Green polygon
        mask = frame.astype(bool)
        img[mask] = cv2.addWeighted(img, 0.5, frame, 0.5, 0)[mask]
    return check_flag



def EstimateCurvatureFromTrajectory(
    traj, min_segment_length: float = 0.05, max_abs_curvature: float = 0.5
):
    """
    根据离散轨迹估计曲率，并对静止/缓慢移动时的数值进行稳定化处理。

    Args:
        traj (np.ndarray): 形状 (N, 3) 或 (N, 2) 的轨迹点。
        min_segment_length (float): 判定“有效位移”的阈值（单位：米）。
        max_abs_curvature (float): 物理可接受的最大曲率，用于裁剪噪声。
    """
    traj = np.asarray(traj)[:, :2]
    curvature = np.zeros(len(traj))

    for i in range(1, len(traj) - 1):
        x1, y1 = traj[i - 1]
        x2, y2 = traj[i]
        x3, y3 = traj[i + 1]

        # Vectors
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x3 - x2, y3 - y2])

        # Lengths
        L1 = np.linalg.norm(v1)
        L2 = np.linalg.norm(v2)
        L3 = np.linalg.norm(np.array([x3 - x1, y3 - y1]))

        # 当车辆几乎静止或噪声导致位移极小时，直接认为曲率为 0
        if (
            L1 < min_segment_length
            or L2 < min_segment_length
            or L3 < min_segment_length * 2
        ):
            curvature[i] = 0.0
            continue

        denom = L1 * L2 * L3
        if denom < 1e-3:
            curvature[i] = 0.0
            continue

        # Signed area (using cross product)
        area_signed = 0.5 * ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        raw_curvature = 4 * area_signed / denom
        curvature[i] = np.clip(raw_curvature, -max_abs_curvature, max_abs_curvature)

    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]

    return curvature

def IntegrateCurvatureForPoints(curvatures, velocities_norm, initial_position, initial_heading, time_span):
    t = np.linspace(0, time_span, time_span)  # Time vector

    # Initial conditions
    x0, y0 = initial_position[0], initial_position[1]  # Starting position
    theta0 = initial_heading  # Initial orientation (radians)

    # Integrate to compute heading (theta)
    theta = cumulative_trapezoid(curvatures * velocities_norm, t, initial=0)
    theta += theta0  # 手动加上初始角度

    # Compute velocity components
    v_x = velocities_norm * np.cos(theta)
    v_y = velocities_norm * np.sin(theta)

    # Integrate to compute trajectory
    x = cumulative_trapezoid(v_x, t, initial=0)
    y = cumulative_trapezoid(v_y, t, initial=0)
    x += x0  # 手动加上初始位置
    y += y0

    return np.stack((x, y), axis=1)

def WriteImageSequenceToVideo(cam_images_sequence: list, filename):
    assert len(cam_images_sequence) >= 1, "No images to write to video."
    # Save the image sequence as video
    # Define the codec and initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(f"{filename}.mp4", fourcc, fps=2,
                                   frameSize=(cam_images_sequence[0].shape[1], cam_images_sequence[0].shape[0]))

    for img in cam_images_sequence:
        video_writer.write(img)

    # Release the video writer
    video_writer.release()