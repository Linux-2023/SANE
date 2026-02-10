#!/usr/bin/env python3
"""
修复 hppfcl 导入问题的脚本

使用 uv/pip 安装 hpp-fcl 后，hppfcl 目录缺少 __init__.py 文件，
导致无法正确导入。此脚本自动检测并修复该问题。

用法:
    python scripts/fix_hppfcl.py
"""

import os
import sys


def find_hppfcl_dir():
    """查找 hppfcl 的安装目录"""
    # 常见的安装路径模式
    patterns = [
        # uv/pip 安装到 cmeel.prefix
        ".venv/lib/python{v}/site-packages/cmeel.prefix/lib/python{v}/site-packages/hppfcl",
        # 标准 site-packages
        ".venv/lib/python{v}/site-packages/hppfcl",
    ]
    
    # 获取 Python 版本
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    for pattern in patterns:
        path = pattern.format(v=py_version)
        if os.path.isdir(path):
            # 检查是否有 .so 文件
            so_files = [f for f in os.listdir(path) if f.endswith('.so')]
            if so_files:
                return path, so_files[0]
    
    return None, None


def fix_hppfcl():
    """修复 hppfcl 导入问题"""
    hppfcl_dir, so_file = find_hppfcl_dir()
    
    if hppfcl_dir is None:
        print("错误: 未找到 hppfcl 安装目录")
        print("请确保已安装 hpp-fcl: uv pip install hpp-fcl")
        return False
    
    init_file = os.path.join(hppfcl_dir, "__init__.py")
    
    # 检查是否已存在
    if os.path.exists(init_file):
        with open(init_file, 'r') as f:
            content = f.read()
        if "from .hppfcl import *" in content:
            print(f"✓ hppfcl 已正确配置: {init_file}")
            return True
    
    # 获取模块名（去掉 .cpython-xx-xxx.so 后缀）
    module_name = so_file.split('.')[0]  # 例如 "hppfcl"
    
    # 创建 __init__.py
    init_content = f"from .{module_name} import *\n"
    
    try:
        with open(init_file, 'w') as f:
            f.write(init_content)
        print(f"✓ 已创建: {init_file}")
        return True
    except PermissionError:
        print(f"错误: 无权限写入 {init_file}")
        print(f"请手动执行: echo 'from .{module_name} import *' > {init_file}")
        return False


def verify_installation():
    """验证安装是否成功"""
    try:
        import pinocchio as pin
        import hppfcl
        
        print(f"\n验证结果:")
        print(f"  pinocchio 版本: {pin.__version__}")
        print(f"  hppfcl.Capsule: {'✓' if hasattr(hppfcl, 'Capsule') else '✗'}")
        print(f"  hppfcl.Sphere:  {'✓' if hasattr(hppfcl, 'Sphere') else '✗'}")
        print(f"  hppfcl.Box:     {'✓' if hasattr(hppfcl, 'Box') else '✗'}")
        
        return hasattr(hppfcl, 'Capsule')
    except ImportError as e:
        print(f"\n验证失败: {e}")
        return False


if __name__ == "__main__":
    print("修复 hppfcl 导入问题...")
    print()
    
    if fix_hppfcl():
        verify_installation()
    else:
        sys.exit(1)

