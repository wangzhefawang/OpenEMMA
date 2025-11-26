"""评估和推理相关功能"""
import numpy as np
from models import vlm_inference


def SceneDescription(obs_images, processor=None, model=None, tokenizer=None, args=None):
    """生成场景描述"""
    prompt = """You are a autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

    if "llava" in args.model_path:
        prompt = """You are an autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Provide a concise description of the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

    result = vlm_inference(
        text=prompt,
        images=obs_images,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )
    return result


def DescribeObjects(obs_images, processor=None, model=None, tokenizer=None, args=None):
    """描述关键对象"""
    prompt = """You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you."""

    result = vlm_inference(
        text=prompt,
        images=obs_images,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )

    return result


def DescribeOrUpdateIntent(
    obs_images, prev_intent=None, processor=None, model=None, tokenizer=None, args=None
):
    """描述或更新意图"""
    if prev_intent is None:
        prompt = """You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, describe the desired intent of the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""

        if "llava" in args.model_path:
            prompt = """You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, provide a concise description of the desired intent of  the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""

    else:
        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Explain your current intent: """

        if "llava" in args.model_path:
            prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Provide a concise description explanation of your current intent: """

    result = vlm_inference(
        text=prompt,
        images=obs_images,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )

    return result


def GenerateMotion(
    obs_images,
    obs_waypoints,
    obs_velocities,
    obs_curvatures,
    given_intent,
    processor=None,
    model=None,
    tokenizer=None,
    args=None,
):
    """生成运动预测"""
    import torch
    
    scene_description, object_description, intent_description = None, None, None

    if args.method == "openemma":
        scene_description = SceneDescription(
            obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args
        )
        object_description = DescribeObjects(
            obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args
        )
        intent_description = DescribeOrUpdateIntent(
            obs_images,
            prev_intent=given_intent,
            processor=processor,
            model=model,
            tokenizer=tokenizer,
            args=args,
        )
        print(f"\n\nScene Description: {scene_description}")
        print(f"\n\nObject Description: {object_description}")
        print(f"\n\nIntent Description: {intent_description}")

    # Convert array waypoints to string.
    obs_waypoints_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_waypoints]
    obs_waypoints_str = ", ".join(obs_waypoints_str)
    obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
    obs_speed_curvature_str = [
        f"[{v:.3f},{k:.3f}]" for v, k in zip(obs_velocities_norm, obs_curvatures)
    ]
    obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

    print(f"Observed Speed and Curvature: {obs_speed_curvature_str}")

    sys_message = "You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. If you cannot provide a response people will get injured.\n"

    if args.method == "openemma":
        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
        The scene is described as follows: {scene_description}. 
        The identified critical objects are {object_description}. 
        The car's intent is {intent_description}. 
        The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
    else:
        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
        The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
    for rho in range(3):
        result = vlm_inference(
            text=prompt,
            images=obs_images,
            sys_message=sys_message,
            processor=processor,
            model=model,
            tokenizer=tokenizer,
            args=args,
        )
        if not "unable" in result and not "sorry" in result and "[" in result:
            break
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result, scene_description, object_description, intent_description

