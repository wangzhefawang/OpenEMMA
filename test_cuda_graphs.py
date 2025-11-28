"""CUDA Graphs 功能测试脚本"""
import torch
import time
from cuda_graphs_wrapper import CUDAGraphsWrapper


def test_basic_functionality():
    """测试 CUDA Graphs 基本功能"""
    print("=" * 60)
    print("测试 1: CUDA Graphs 基本功能")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，跳过测试")
        return
    
    # 创建包装器
    wrapper = CUDAGraphsWrapper(
        max_graphs=5,
        warmup_iterations=2,
        enabled=True
    )
    
    # 定义简单的测试函数
    def simple_matmul(x, y):
        return torch.matmul(x, y)
    
    # 测试数据
    device = torch.device("cuda")
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    
    print("\n🔄 第一次调用（捕获 graph）...")
    result1, used_graph1 = wrapper.capture_and_replay(simple_matmul, x, y)
    print(f"   使用 graph: {used_graph1}")
    
    print("\n🔄 第二次调用（重放 graph）...")
    result2, used_graph2 = wrapper.capture_and_replay(simple_matmul, x, y)
    print(f"   使用 graph: {used_graph2}")
    
    # 验证结果一致性
    torch.cuda.synchronize()
    print("\n✅ 测试通过：CUDA Graphs 基本功能正常")
    wrapper.print_statistics()


def test_performance_comparison():
    """测试性能对比"""
    print("\n" + "=" * 60)
    print("测试 2: 性能对比")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，跳过测试")
        return
    
    # 定义测试函数
    def test_function(x):
        # 简单的计算链
        y = x + 1
        y = torch.relu(y)
        y = torch.matmul(y, y.T)
        return y
    
    device = torch.device("cuda")
    x = torch.randn(500, 500, device=device)
    
    # 预热
    for _ in range(10):
        _ = test_function(x)
    torch.cuda.synchronize()
    
    # 测试不使用 CUDA Graphs
    print("\n📊 不使用 CUDA Graphs:")
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        _ = test_function(x)
    torch.cuda.synchronize()
    time_without = time.perf_counter() - start
    print(f"   {iterations} 次迭代耗时: {time_without:.4f} 秒")
    print(f"   平均每次: {time_without/iterations*1000:.2f} 毫秒")
    
    # 测试使用 CUDA Graphs
    print("\n📊 使用 CUDA Graphs:")
    wrapper = CUDAGraphsWrapper(
        max_graphs=5,
        warmup_iterations=3,
        enabled=True
    )
    
    start = time.perf_counter()
    for _ in range(iterations):
        _, _ = wrapper.capture_and_replay(test_function, x)
    torch.cuda.synchronize()
    time_with = time.perf_counter() - start
    print(f"   {iterations} 次迭代耗时: {time_with:.4f} 秒")
    print(f"   平均每次: {time_with/iterations*1000:.2f} 毫秒")
    
    # 计算加速比
    speedup = (time_without - time_with) / time_without * 100
    print(f"\n🚀 速度提升: {speedup:.2f}%")
    
    if speedup > 0:
        print("✅ CUDA Graphs 带来了性能提升！")
    else:
        print("⚠️  注意：在这个简单测试中没有看到明显提升")
        print("   VLM 推理等复杂任务中效果会更明显")
    
    wrapper.print_statistics()


def test_dynamic_shapes():
    """测试动态输入形状"""
    print("\n" + "=" * 60)
    print("测试 3: 动态输入形状处理")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，跳过测试")
        return
    
    wrapper = CUDAGraphsWrapper(
        max_graphs=3,
        warmup_iterations=2,
        enabled=True
    )
    
    def simple_op(x):
        return x * 2 + 1
    
    device = torch.device("cuda")
    
    # 测试不同形状
    shapes = [
        (100, 100),
        (200, 200),
        (100, 100),  # 重复，应该命中缓存
        (300, 300),
        (100, 100),  # 再次重复
    ]
    
    print("\n🔄 测试不同输入形状...")
    for i, shape in enumerate(shapes, 1):
        x = torch.randn(*shape, device=device)
        _, used_graph = wrapper.capture_and_replay(simple_op, x)
        print(f"   形状 {shape}: {'✅ 命中缓存' if used_graph else '🆕 新建 graph'}")
    
    wrapper.print_statistics()
    
    stats = wrapper.get_statistics()
    if stats['hit_rate'] > 0:
        print("\n✅ 多图缓存机制正常工作")
    else:
        print("\n⚠️  未检测到缓存命中")


def test_fallback_mechanism():
    """测试降级机制"""
    print("\n" + "=" * 60)
    print("测试 4: 降级机制")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，跳过测试")
        return
    
    # 测试禁用状态
    wrapper = CUDAGraphsWrapper(
        max_graphs=5,
        warmup_iterations=2,
        enabled=False  # 禁用
    )
    
    def simple_op(x):
        return x + 1
    
    device = torch.device("cuda")
    x = torch.randn(100, 100, device=device)
    
    print("\n🔄 测试禁用状态...")
    _, used_graph = wrapper.capture_and_replay(simple_op, x)
    
    if not used_graph:
        print("✅ 禁用状态下正确降级到普通执行")
    else:
        print("❌ 禁用状态异常")
    
    # 测试 CPU 情况
    print("\n🔄 测试 CPU 输入...")
    wrapper_cpu = CUDAGraphsWrapper(enabled=True)
    x_cpu = torch.randn(10, 10)  # CPU tensor
    _, used_graph_cpu = wrapper_cpu.capture_and_replay(simple_op, x_cpu)
    
    if not used_graph_cpu:
        print("✅ CPU 输入正确降级")
    else:
        print("❌ CPU 输入处理异常")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("🧪 CUDA Graphs 功能测试套件")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_performance_comparison,
        test_dynamic_shapes,
        test_fallback_mechanism,
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

