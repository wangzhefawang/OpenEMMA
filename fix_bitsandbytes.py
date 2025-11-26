"""
自动修复 bitsandbytes 8bit 量化 bug 的脚本

使用方法：
    python fix_bitsandbytes.py
"""
import sys
import io
from pathlib import Path

# 设置输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def find_bitsandbytes_file():
    """查找 bitsandbytes 安装位置"""
    try:
        import bitsandbytes
        bnb_path = Path(bitsandbytes.__file__).parent
        target_file = bnb_path / "backends" / "cuda" / "ops.py"
        return target_file
    except ImportError:
        print("[ERROR] 未找到 bitsandbytes 库，请先安装：pip install bitsandbytes")
        return None

def fix_view_bug(file_path):
    """修复 .view(-1) 为 .reshape(-1)"""
    if not file_path.exists():
        print(f"[ERROR] 文件不存在: {file_path}")
        return False
    
    # 读取文件
    content = file_path.read_text(encoding='utf-8')
    original_content = content
    
    # 查找并替换
    old_line = "outlier_cols = torch.argwhere(outliers.any(dim=0)).view(-1)"
    new_line = "outlier_cols = torch.argwhere(outliers.any(dim=0)).reshape(-1)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # 备份原文件
        backup_path = file_path.with_suffix('.py.backup')
        backup_path.write_text(original_content, encoding='utf-8')
        print(f"[OK] 已备份原文件到: {backup_path}")
        
        # 写入修复后的内容
        file_path.write_text(content, encoding='utf-8')
        print(f"[OK] 已修复文件: {file_path}")
        print(f"     修改: .view(-1) -> .reshape(-1)")
        return True
    elif new_line in content:
        print(f"[INFO] 文件已经修复过了，无需再次修复")
        return True
    else:
        print(f"[WARN] 未找到需要修复的代码行")
        return False

def main():
    print("=" * 60)
    print("bitsandbytes 8bit 量化 Bug 修复工具")
    print("=" * 60)
    
    # 查找文件
    target_file = find_bitsandbytes_file()
    if not target_file:
        sys.exit(1)
    
    print(f"\n[FILE] 找到文件: {target_file}")
    
    # 修复
    if fix_view_bug(target_file):
        print("\n" + "=" * 60)
        print("[SUCCESS] 修复完成！现在可以使用 --quantization 8bit 了")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("[FAILED] 修复失败，请手动修改")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()

