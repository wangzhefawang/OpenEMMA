"""CUDA Graphs 优化包装器 - 减少 Python 调度开销"""
import torch
from typing import Dict, Tuple, Optional, Callable
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CUDAGraphsWrapper:
    """
    CUDA Graphs 包装器，用于加速重复的推理计算
    
    特性：
    - 自动捕获和缓存多个 graph（支持不同输入形状）
    - 智能降级：遇到不支持的操作自动回退
    - LRU 缓存：限制最多缓存的 graph 数量
    """
    
    def __init__(
        self, 
        max_graphs: int = 10,
        warmup_iterations: int = 3,
        enabled: bool = True
    ):
        """
        初始化 CUDA Graphs 包装器
        
        Args:
            max_graphs: 最大缓存的 graph 数量
            warmup_iterations: 预热迭代次数
            enabled: 是否启用 CUDA Graphs
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.max_graphs = max_graphs
        self.warmup_iterations = warmup_iterations
        
        # 存储捕获的 graphs：key 是输入签名，value 是 (graph, static_inputs, static_outputs)
        self.graph_cache: OrderedDict = OrderedDict()
        
        # 统计信息
        self.stats = {
            "graph_hits": 0,
            "graph_misses": 0,
            "fallbacks": 0,
            "total_captures": 0,
        }
        
        if self.enabled:
            logger.info("✅ CUDA Graphs 优化已启用")
            logger.info(f"   - 最大缓存图数: {max_graphs}")
            logger.info(f"   - 预热迭代次数: {warmup_iterations}")
        else:
            logger.info("⚠️  CUDA Graphs 优化未启用")
    
    def get_input_signature(self, *args, **kwargs) -> Optional[str]:
        """
        生成输入参数的签名，用于识别相同的计算图
        
        Returns:
            输入签名字符串，如果无法生成则返回 None
        """
        try:
            sig_parts = []
            
            # 处理位置参数
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    sig_parts.append(f"T{tuple(arg.shape)}_{arg.dtype}_{arg.device}")
                elif isinstance(arg, (int, float, str, bool)):
                    sig_parts.append(f"V{type(arg).__name__}_{arg}")
            
            # 处理关键字参数（只包含重要的参数）
            important_kwargs = ["max_new_tokens", "temperature", "do_sample", "num_beams"]
            for key in important_kwargs:
                if key in kwargs:
                    val = kwargs[key]
                    sig_parts.append(f"K{key}_{val}")
            
            return "|".join(sig_parts)
        except Exception as e:
            logger.warning(f"无法生成输入签名: {e}")
            return None
    
    def _clone_tensors(self, data):
        """深度克隆张量数据结构"""
        if isinstance(data, torch.Tensor):
            return data.clone()
        elif isinstance(data, dict):
            return {k: self._clone_tensors(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._clone_tensors(item) for item in data)
        else:
            return data
    
    def _copy_tensors(self, src, dst):
        """将 src 的张量数据复制到 dst（in-place）"""
        if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
            dst.copy_(src)
        elif isinstance(src, dict) and isinstance(dst, dict):
            for k in src.keys():
                if k in dst:
                    self._copy_tensors(src[k], dst[k])
        elif isinstance(src, (list, tuple)) and isinstance(dst, (list, tuple)):
            for s, d in zip(src, dst):
                self._copy_tensors(s, d)
    
    def _check_all_cuda(self, *args, **kwargs) -> bool:
        """检查所有输入张量是否都在 CUDA 设备上"""
        def check_item(item):
            if isinstance(item, torch.Tensor):
                return item.is_cuda
            elif isinstance(item, (list, tuple)):
                return all(check_item(x) for x in item)
            elif isinstance(item, dict):
                return all(check_item(v) for v in item.values())
            return True
        
        # 检查位置参数
        for arg in args:
            if not check_item(arg):
                return False
        
        # 检查关键字参数
        for val in kwargs.values():
            if not check_item(val):
                return False
        
        return True
    
    def capture_and_replay(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Tuple[any, bool]:
        """
        捕获或重放 CUDA Graph
        
        Args:
            func: 要包装的函数
            *args, **kwargs: 函数参数
            
        Returns:
            (output, used_graph): 函数输出和是否使用了 graph
        """
        if not self.enabled:
            return func(*args, **kwargs), False
        
        # 检查所有输入是否都在 CUDA 上
        if not self._check_all_cuda(*args, **kwargs):
            self.stats["fallbacks"] += 1
            return func(*args, **kwargs), False
        
        # 生成输入签名
        signature = self.get_input_signature(*args, **kwargs)
        if signature is None:
            self.stats["fallbacks"] += 1
            return func(*args, **kwargs), False
        
        # 检查缓存
        if signature in self.graph_cache:
            # 命中缓存，重放 graph
            graph, static_inputs, static_outputs = self.graph_cache[signature]
            
            # 将新输入复制到静态缓冲区
            self._copy_tensors(args, static_inputs["args"])
            for key in kwargs:
                if key in static_inputs["kwargs"]:
                    self._copy_tensors(kwargs[key], static_inputs["kwargs"][key])
            
            # 重放 graph
            graph.replay()
            
            # 移动到 LRU 队列末尾
            self.graph_cache.move_to_end(signature)
            self.stats["graph_hits"] += 1
            
            # 返回输出（克隆以避免被覆盖）
            return self._clone_tensors(static_outputs), True
        
        # 未命中缓存，需要捕获新 graph
        self.stats["graph_misses"] += 1
        
        try:
            # 创建静态输入缓冲区
            static_args = self._clone_tensors(args)
            static_kwargs = self._clone_tensors(kwargs)
            
            # 预热
            with torch.cuda.stream(torch.cuda.Stream()):
                for _ in range(self.warmup_iterations):
                    _ = func(*static_args, **static_kwargs)
            torch.cuda.synchronize()
            
            # 捕获 graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_outputs = func(*static_args, **static_kwargs)
            
            # 缓存 graph
            self.graph_cache[signature] = (
                graph,
                {"args": static_args, "kwargs": static_kwargs},
                static_outputs
            )
            self.stats["total_captures"] += 1
            
            # 限制缓存大小（LRU 淘汰）
            if len(self.graph_cache) > self.max_graphs:
                removed_key = next(iter(self.graph_cache))
                del self.graph_cache[removed_key]
                logger.info(f"LRU 淘汰旧 graph，当前缓存: {len(self.graph_cache)}/{self.max_graphs}")
            
            logger.info(f"✨ 捕获新 CUDA Graph (总数: {len(self.graph_cache)})")
            
            # 首次执行：将实际输入复制到静态缓冲区并重放
            self._copy_tensors(args, static_args)
            for key in kwargs:
                if key in static_kwargs:
                    self._copy_tensors(kwargs[key], static_kwargs[key])
            
            graph.replay()
            return self._clone_tensors(static_outputs), True
            
        except Exception as e:
            # 捕获失败，降级到普通执行
            logger.warning(f"CUDA Graph 捕获失败，降级到普通执行: {e}")
            self.stats["fallbacks"] += 1
            return func(*args, **kwargs), False
    
    def get_statistics(self) -> Dict[str, any]:
        """获取统计信息"""
        total_calls = self.stats["graph_hits"] + self.stats["graph_misses"]
        hit_rate = self.stats["graph_hits"] / total_calls if total_calls > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cached_graphs": len(self.graph_cache),
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("CUDA Graphs 统计信息")
        print("=" * 60)
        print(f"缓存命中次数: {stats['graph_hits']}")
        print(f"缓存未命中次数: {stats['graph_misses']}")
        print(f"命中率: {stats['hit_rate']:.2%}")
        print(f"降级执行次数: {stats['fallbacks']}")
        print(f"总捕获图数: {stats['total_captures']}")
        print(f"当前缓存图数: {stats['cached_graphs']}/{self.max_graphs}")
        print("=" * 60 + "\n")
    
    def clear_cache(self):
        """清空缓存"""
        self.graph_cache.clear()
        logger.info("已清空 CUDA Graphs 缓存")


class SimpleCUDAGraphWrapper:
    """
    简化版 CUDA Graph 包装器，用于不支持动态输入的场景
    只捕获一个 graph，要求所有输入形状完全一致
    """
    
    def __init__(self, warmup_iterations: int = 3):
        self.warmup_iterations = warmup_iterations
        self.graph = None
        self.static_inputs = None
        self.static_outputs = None
        self.initialized = False
        
    def __call__(self, func: Callable, *args, **kwargs):
        """
        执行函数，首次调用时捕获 graph
        """
        if not torch.cuda.is_available():
            return func(*args, **kwargs)
        
        if not self.initialized:
            # 首次初始化
            logger.info("🔄 初始化 CUDA Graph（简化版）...")
            
            # 预热
            for _ in range(self.warmup_iterations):
                _ = func(*args, **kwargs)
            torch.cuda.synchronize()
            
            # 捕获
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_outputs = func(*args, **kwargs)
            
            self.initialized = True
            logger.info("✅ CUDA Graph 捕获完成")
        
        # 重放
        self.graph.replay()
        return self.static_outputs


# 全局实例（可选）
_global_wrapper: Optional[CUDAGraphsWrapper] = None


def get_cuda_graphs_wrapper(
    max_graphs: int = 10,
    warmup_iterations: int = 3,
    enabled: bool = True
) -> CUDAGraphsWrapper:
    """获取或创建全局 CUDA Graphs 包装器"""
    global _global_wrapper
    
    if _global_wrapper is None:
        _global_wrapper = CUDAGraphsWrapper(
            max_graphs=max_graphs,
            warmup_iterations=warmup_iterations,
            enabled=enabled
        )
    
    return _global_wrapper

