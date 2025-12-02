"""模型加载和推理相关功能"""
import re
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from llava.model.builder import load_pretrained_model
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.conversation import conv_templates
from openai import OpenAI
from cuda_graphs_wrapper import CUDAGraphsWrapper

client = OpenAI(api_key="[your-openai-api-key]")

# 全局 CUDA Graphs 包装器实例
_cuda_graphs_wrapper = None


def initialize_cuda_graphs(use_cuda_graphs: bool = False, warmup_iterations: int = 3):
    """
    初始化全局 CUDA Graphs 包装器
    
    Args:
        use_cuda_graphs: 是否启用 CUDA Graphs
        warmup_iterations: 预热迭代次数
    """
    global _cuda_graphs_wrapper
    
    if _cuda_graphs_wrapper is None and use_cuda_graphs:
        _cuda_graphs_wrapper = CUDAGraphsWrapper(
            max_graphs=10,
            warmup_iterations=warmup_iterations,
            enabled=use_cuda_graphs and torch.cuda.is_available()
        )
    
    return _cuda_graphs_wrapper


def get_cuda_graphs_wrapper():
    """获取全局 CUDA Graphs 包装器"""
    return _cuda_graphs_wrapper


def load_vlm(repo_or_path, quantization: str = "none", use_cuda_graphs: bool = False, warmup_iterations: int = 3):
    """
    根据 --model-path 加载对应的视觉语言模型。
    - 对于 LLaVA 系列，走其自带的加载逻辑（需要 tokenizer / image_processor）。
    - 其他 Hugging Face Vision-LLM 走 AutoModelForVision2Seq + AutoProcessor。
    
    Args:
        repo_or_path: 模型路径或仓库名
        quantization: 量化方式 (none/4bit/8bit)
        use_cuda_graphs: 是否启用 CUDA Graphs 优化
        warmup_iterations: CUDA Graphs 预热迭代次数
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_lower = repo_or_path.lower()
    quantization = (quantization or "none").lower()
    
    # 初始化 CUDA Graphs 包装器
    if use_cuda_graphs:
        initialize_cuda_graphs(use_cuda_graphs, warmup_iterations)

    if "llava" in repo_lower:
        disable_torch_init()
        model_name = get_model_name_from_path(repo_or_path)
        load_kwargs = dict(
            model_path=repo_or_path,
            model_base=None,
            model_name=model_name,
            device=device,
            device_map="auto" if device == "cuda" else {"": device},
            load_4bit=quantization == "4bit",
            load_8bit=quantization == "8bit",
        )
        try:
            tokenizer, model, image_processor, _ = load_pretrained_model(**load_kwargs)
        except Exception as err:
            if quantization != "none":
                print(f"量化加载失败（{err}），回退到全精度……")
                load_kwargs["load_4bit"] = False
                load_kwargs["load_8bit"] = False
                tokenizer, model, image_processor, _ = load_pretrained_model(**load_kwargs)
            else:
                raise
        if device != "cuda":
            model = model.to(device)
        return model, tokenizer, image_processor

    repo = repo_or_path  # 既可本地目录，也可 HF 仓库名
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    try:
        proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    except Exception:
        proc = None
    quant_config = None
    if quantization in {"4bit", "8bit"}:
        if not torch.cuda.is_available():
            print("警告：当前设备不支持量化加载，改为默认精度。")
            quantization = "none"
        else:
            if quantization == "4bit":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif quantization == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)

    def _load_default_precision():
        mdl = AutoModelForVision2Seq.from_pretrained(
            repo, trust_remote_code=True, torch_dtype="auto"
        )
        return mdl.to(device)

    if quant_config is not None:
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                repo,
                trust_remote_code=True,
                quantization_config=quant_config,
                device_map="auto",
            )
        except Exception as err:
            print(f"量化加载失败（{err}），回退到全精度……")
            model = _load_default_precision()
    else:
        model = _load_default_precision()
    return model, tok, proc


def getMessage(prompt, image=None, args=None):
    """构建不同模型的消息格式"""
    if "llama" in args.model_path or "Llama" in args.model_path:
        message = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
    elif "qwen" in args.model_path or "Qwen" in args.model_path:
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    return message


def prepare_image_payload(image_source, args=None, processor=None, model=None):
    """
    为多次调用的 VLM 推理准备图像缓存，避免重复磁盘 IO 和张量预处理。
    仅在 LLaVA/LLaMA 路线下创建 payload，其它模型保持原样返回。
    """
    if args is None:
        return image_source
    model_lower = args.model_path.lower()
    if not any(key in model_lower for key in ("llava", "llama")):
        return image_source

    if isinstance(image_source, dict) and "pil" in image_source:
        return image_source

    if isinstance(image_source, Image.Image):
        pil_image = image_source
    else:
        pil_image = Image.open(image_source).convert("RGB")
    payload = {"pil": pil_image}

    if "llava" in model_lower and processor is not None and model is not None:
        # 只在 CPU 上缓存一次 processed tensor，使用时再拷贝到 GPU
        with torch.inference_mode():
            payload["llava_tensor"] = (
                process_images([pil_image], processor, model.config)[0].cpu()
            )

    return payload


def vlm_inference(
    text=None,
    images=None,
    sys_message=None,
    processor=None,
    model=None,
    tokenizer=None,
    args=None,
    use_cuda_graphs=False,
):
    """
    统一的 VLM 推理接口
    
    Args:
        text: 输入文本
        images: 输入图像
        sys_message: 系统消息
        processor: 处理器
        model: 模型
        tokenizer: 分词器
        args: 命令行参数
        use_cuda_graphs: 是否使用 CUDA Graphs（实验性功能）
    
    Note:
        CUDA Graphs 对于具有动态输出长度的生成任务支持有限。
        当前主要优化输入处理和编码阶段。
    """
    cuda_wrapper = get_cuda_graphs_wrapper() if use_cuda_graphs else None
    
    with torch.inference_mode():
        if "llama" in args.model_path or "Llama" in args.model_path:
            if isinstance(images, dict) and "pil" in images:
                image = images["pil"]
            else:
                image = Image.open(images).convert("RGB")
            message = getMessage(text, args=args)
            input_text = processor.apply_chat_template(
                message, add_generation_prompt=True
            )
            inputs = processor(
                image, input_text, add_special_tokens=False, return_tensors="pt"
            ).to(model.device)

            max_tokens = getattr(args, 'max_new_tokens', None) or 2048  # Llama 默认 2048
            output = model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

            output_text = processor.decode(output[0])
            
            # 立即释放显存
            del inputs, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if "llama" in args.model_path or "Llama" in args.model_path:
                # 尝试提取 assistant 的回复（支持多种格式）
                # 格式1: assistant 回复后有 <|eot_id|>
                matches = re.findall(
                    r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*(.*?)(?:<\|eot_id\|>|$)",
                    output_text,
                    re.DOTALL,
                )
                if matches:
                    # 取最后一个 assistant 的回复（可能有多轮对话）
                    output_text = matches[-1].strip()
                else:
                    # 如果正则匹配失败，尝试其他格式或直接返回
                    print(f"[WARNING] 无法解析 Llama 输出格式，使用备用方案")
                    
                    # 尝试简单清理：移除特殊 token
                    output_text = output_text.replace("<|begin_of_text|>", "")
                    output_text = output_text.replace("<|end_of_text|>", "")
                    output_text = output_text.replace("<|eot_id|>", "")
                    
                    # 尝试提取 assistant 后的内容
                    if "assistant" in output_text:
                        parts = output_text.split("assistant", 1)
                        if len(parts) > 1:
                            output_text = parts[1].strip()
                    
                    output_text = output_text.strip()
            return output_text

        elif "qwen" in args.model_path or "Qwen" in args.model_path:
            # 判断是否为Qwen2.5-VL-3B-Instruct（新版）
            if hasattr(model, "model_type") and getattr(model, "model_type", "") == "qwen2_5_vl":
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": images},
                            {"type": "text", "text": text}
                        ]
                    }
                ]
                text_prompt = processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(message)
                inputs = processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)
                max_tokens = getattr(args, 'max_new_tokens', None) or 512  # Qwen2.5-VL 默认 512
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                return output_text[0]
            else:
                # 兼容Qwen2-VL-7B-Instruct等老模型
                message = getMessage(text, image=images, args=args)
                text_prompt = processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(message)
                inputs = processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
                max_tokens = getattr(args, 'max_new_tokens', None) or 512  # Qwen2-VL 默认 512
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                # 释放显存
                del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return output_text[0]

        elif "llava" in args.model_path:
            conv_mode = "mistral_instruct"
            image_token_se = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            if IMAGE_PLACEHOLDER in text:
                if model.config.mm_use_im_start_end:
                    text = re.sub(IMAGE_PLACEHOLDER, image_token_se, text)
                else:
                    text = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, text)
            else:
                if model.config.mm_use_im_start_end:
                    text = image_token_se + "\n" + text
                else:
                    text = DEFAULT_IMAGE_TOKEN + "\n" + text

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .to(model.device)
            )
            if isinstance(images, dict) and "pil" in images:
                image = images["pil"]
                image_tensor = images.get("llava_tensor")
            else:
                image = Image.open(images).convert("RGB")
                image_tensor = None

            if image_tensor is None:
                image_tensor = process_images([image], processor, model.config)[0].cpu()

            image_tensor = image_tensor.unsqueeze(0).to(model.device, dtype=torch.float16)

            max_tokens = getattr(args, 'max_new_tokens', None) or 2048  # LLaVA 默认 2048
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                max_new_tokens=max_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # 释放显存
            del input_ids, image_tensor, output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return outputs

        elif "gpt" in args.model_path:
            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        *map(lambda x: {"image": x, "resize": 768}, images),
                        text,
                    ],
                },
            ]
            if sys_message is not None:
                sys_message_dict = {"role": "system", "content": sys_message}
                PROMPT_MESSAGES.append(sys_message_dict)
            max_tokens = getattr(args, 'max_new_tokens', None) or 800  # GPT 默认 800
            params = {
                "model": "gpt-4o-2024-11-20",
                "messages": PROMPT_MESSAGES,
                "max_tokens": max_tokens,
            }

            result = client.chat.completions.create(**params)

            return result.choices[0].message.content

