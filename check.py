import torch, time

print("Torch:", torch.__version__)
print("CUDA in wheel:", torch.version.cuda)

if torch.cuda.is_available():
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print("Device:", dev)
    print("Capability:", cap)
    # 简单算一次 GEMM 看是否走到 GPU 快路径
    x = torch.randn(2048, 2048, device='cuda')
    torch.cuda.synchronize()
    t0 = time.time()
    y = x @ x
    torch.cuda.synchronize()
    print("GEMM ms:", (time.time() - t0) * 1000)
else:
    print("GPU not available")
