"""格式化工具函数 - 用于统一显示格式"""


def format_runtime(seconds: float) -> str:
    """
    格式化运行时间为易读格式
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化后的字符串，如 "2小时 30分钟 45秒"
        
    Examples:
        >>> format_runtime(45)
        '45秒'
        >>> format_runtime(150)
        '2分钟 30秒'
        >>> format_runtime(7265)
        '2小时 1分钟 5秒'
    """
    if seconds is None:
        return "N/A"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}小时 {minutes}分钟 {secs}秒"
    elif minutes > 0:
        return f"{minutes}分钟 {secs}秒"
    else:
        return f"{secs}秒"


def format_runtime_with_raw(seconds: float) -> str:
    """
    格式化运行时间，同时显示原始秒数
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化后的字符串，如 "2小时 30分钟 45秒 (9045.00秒)"
    """
    if seconds is None:
        return "N/A"
    
    formatted = format_runtime(seconds)
    return f"{formatted} ({seconds:.2f}秒)"


def format_memory(mb: float) -> str:
    """
    格式化内存/显存大小为易读格式
    
    Args:
        mb: MB 数量
        
    Returns:
        格式化后的字符串，如 "10.24 GB" 或 "512.50 MB"
        
    Examples:
        >>> format_memory(512)
        '512.00 MB'
        >>> format_memory(2048)
        '2.00 GB'
        >>> format_memory(10240)
        '10.00 GB'
    """
    if mb is None:
        return "N/A"
    
    if mb >= 1024:
        gb = mb / 1024
        return f"{gb:.2f} GB"
    else:
        return f"{mb:.2f} MB"


def format_memory_with_raw(mb: float) -> str:
    """
    格式化内存/显存大小，同时显示原始 MB 数
    
    Args:
        mb: MB 数量
        
    Returns:
        格式化后的字符串，如 "10.24 GB (10485.76 MB)"
    """
    if mb is None:
        return "N/A"
    
    if mb >= 1024:
        gb = mb / 1024
        return f"{gb:.2f} GB ({mb:.2f} MB)"
    else:
        return f"{mb:.2f} MB"


def format_bytes(bytes_val: float) -> str:
    """
    格式化字节数为易读格式
    
    Args:
        bytes_val: 字节数
        
    Returns:
        格式化后的字符串，如 "1.50 GB" 或 "512.00 KB"
        
    Examples:
        >>> format_bytes(1024)
        '1.00 KB'
        >>> format_bytes(1048576)
        '1.00 MB'
        >>> format_bytes(1073741824)
        '1.00 GB'
    """
    if bytes_val is None:
        return "N/A"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def format_percentage(value: float, total: float) -> str:
    """
    格式化百分比
    
    Args:
        value: 当前值
        total: 总值
        
    Returns:
        格式化后的字符串，如 "75.50%"
    """
    if value is None or total is None or total == 0:
        return "N/A"
    
    percentage = (value / total) * 100
    return f"{percentage:.1f}%"


if __name__ == "__main__":
    # 测试
    print("运行时间格式化测试：")
    print(f"  45秒: {format_runtime(45)}")
    print(f"  150秒: {format_runtime(150)}")
    print(f"  7265秒: {format_runtime(7265)}")
    print(f"  7265秒(带原始): {format_runtime_with_raw(7265)}")
    
    print("\n内存格式化测试：")
    print(f"  512 MB: {format_memory(512)}")
    print(f"  2048 MB: {format_memory(2048)}")
    print(f"  10240 MB: {format_memory(10240)}")
    print(f"  10240 MB(带原始): {format_memory_with_raw(10240)}")
    
    print("\n字节格式化测试：")
    print(f"  1024 bytes: {format_bytes(1024)}")
    print(f"  1048576 bytes: {format_bytes(1048576)}")
    print(f"  1073741824 bytes: {format_bytes(1073741824)}")

