"""
实时监控 GPU 显存使用情况

使用方法：
    python monitor_gpu.py
"""
import time
import torch


def format_bytes(bytes_val):
    """格式化字节数"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def monitor_gpu(interval=2.0):
    """持续监控 GPU 显存"""
    if not torch.cuda.is_available():
        print("CUDA 不可用，无法监控 GPU")
        return
    
    device_count = torch.cuda.device_count()
    print(f"检测到 {device_count} 个 GPU 设备")
    print("=" * 80)
    
    try:
        while True:
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = props.total_memory
                
                allocated_pct = (allocated / total) * 100
                reserved_pct = (reserved / total) * 100
                
                print(f"\nGPU {i}: {props.name}")
                print(f"  已分配: {format_bytes(allocated)} / {format_bytes(total)} ({allocated_pct:.1f}%)")
                print(f"  已保留: {format_bytes(reserved)} / {format_bytes(total)} ({reserved_pct:.1f}%)")
                print(f"  可用:   {format_bytes(total - reserved)}")
                
                # 警告
                if allocated_pct > 90:
                    print(f"  ⚠️  警告：显存使用率超过 90%！")
                elif allocated_pct > 80:
                    print(f"  ⚠️  注意：显存使用率超过 80%")
            
            print("\n" + "=" * 80)
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="监控 GPU 显存使用")
    parser.add_argument("--interval", type=float, default=2.0, help="刷新间隔（秒）")
    args = parser.parse_args()
    
    monitor_gpu(args.interval)

