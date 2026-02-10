#!/usr/bin/env python3
"""
HDF5文件prompt修改工具

该脚本用于读取HDF5文件，修改其中的prompt（task字段），并生成一个新的HDF5文件，避免直接覆盖原文件。
"""

import h5py
import numpy as np
import argparse
import os
import re
from typing import Optional


def make_safe_filename(prompt: str, max_length: int = 80) -> str:
    """
    将prompt转换为文件名安全的格式
    
    Args:
        prompt: 原始prompt内容
        max_length: 最大长度限制
    
    Returns:
        安全的文件名字符串
    """
    safe_prompt = prompt.replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe_prompt = safe_prompt.replace(",", "").replace(".", "")
    if len(safe_prompt) > max_length:
        safe_prompt = safe_prompt[:max_length]
    return safe_prompt


def extract_timestamp_from_filename(filename: str) -> Optional[str]:
    """
    从文件名中提取时间戳前缀（episode_YYYYMMDD_HHMMSS）
    
    Args:
        filename: 文件名（不含路径）
    
    Returns:
        时间戳前缀，如果无法识别则返回None
    """
    # 匹配 episode_YYYYMMDD_HHMMSS 格式
    pattern = r'^episode_(\d{8}_\d{6})'
    match = re.match(pattern, filename)
    if match:
        return f"episode_{match.group(1)}"
    
    # 如果无法匹配标准格式，尝试提取到第一个下划线后的第二个下划线之前
    # 这适用于 episode_TIMESTAMP_* 格式
    parts = filename.split('_')
    if len(parts) >= 3 and parts[0] == 'episode':
        # 假设时间戳格式是 YYYYMMDD_HHMMSS（两部分）
        if len(parts) >= 3:
            return '_'.join(parts[:3])  # episode_TIMESTAMP1_TIMESTAMP2
    
    return None


def generate_output_path(file_path: str, new_prompt: str = None, 
                         original_prompt: str = None, suffix: str = "_modified") -> str:
    """
    生成新的HDF5输出路径，保留时间前缀，替换prompt部分
    
    Args:
        file_path: 原文件路径
        new_prompt: 新的prompt内容（如果提供，将用于生成文件名）
        original_prompt: 原始prompt内容（用于从文件名中识别和移除）
        suffix: 如果无法解析时间前缀，使用此后缀标识
    
    Returns:
        新文件路径
    """
    directory, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)
    
    if not ext:
        ext = '.hdf5'
    
    # 尝试提取时间戳前缀
    timestamp_prefix = extract_timestamp_from_filename(name)
    
    if timestamp_prefix and new_prompt:
        # 使用时间戳前缀 + 新prompt生成文件名
        safe_prompt = make_safe_filename(new_prompt)
        new_name = f"{timestamp_prefix}_{safe_prompt}"
        candidate = os.path.join(directory, f"{new_name}{ext}")
        
        # 如果文件已存在，添加计数器
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(directory, f"{new_name}_{counter}{ext}")
            counter += 1
        return candidate
    else:
        # 回退到原来的逻辑：在原文件名后添加后缀
        candidate = os.path.join(directory, f"{name}{suffix}{ext}")
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(directory, f"{name}{suffix}_{counter}{ext}")
            counter += 1
        return candidate


def check_hdf5_structure(file_path: str) -> dict:
    """
    检查HDF5文件结构
    
    Args:
        file_path: HDF5文件路径
    
    Returns:
        包含文件信息的字典
    """
    info = {}
    
    with h5py.File(file_path, 'r') as f:
        if 'task' in f:
            task_data = f['task'][:]
            info['has_task'] = True
            info['task_shape'] = task_data.shape
            info['task_dtype'] = task_data.dtype
            info['current_prompt'] = task_data[0].decode('utf-8') if len(task_data) > 0 else ""
            info['unique_prompts'] = list(set([task.decode('utf-8') for task in task_data]))
        else:
            info['has_task'] = False
            # 列出所有可用的字段
            info['available_fields'] = list(f.keys())
    
    return info


def modify_prompt_in_hdf5(file_path: str, new_prompt: str, 
                         prompt_field: str = 'task', 
                         output_file: str = None) -> bool:
    """
    修改HDF5文件中的prompt，并生成新的文件
    
    Args:
        file_path: HDF5文件路径
        new_prompt: 新的prompt内容
        prompt_field: prompt字段名称（默认为'task'）
        output_file: 输出文件路径（默认在原文件同目录追加后缀）
    
    Returns:
        是否生成成功
    """

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return False
    
    try:
        # 首先读取文件信息
        info = check_hdf5_structure(file_path)
        
        if not info.get('has_task', False):
            print(f"错误：文件中没有找到 '{prompt_field}' 字段")
            if 'available_fields' in info:
                print(f"可用字段: {info['available_fields']}")
            return False
        
        # 如果output_file未指定，现在生成（此时已获取原始prompt信息）
        if output_file is None:
            original_prompt = info.get('current_prompt', '')
            output_file = generate_output_path(file_path, new_prompt, original_prompt)
        
        if os.path.abspath(output_file) == os.path.abspath(file_path):
            print("错误：输出文件路径不能与原文件相同")
            return False
        
        print(f"当前prompt: {info['current_prompt']}")
        print(f"新prompt: {new_prompt}")
        print(f"输出文件: {output_file}")
        
        # 以只读模式打开原文件，并创建新文件写入
        with h5py.File(file_path, 'r') as src, h5py.File(output_file, 'w') as dst:
            # 拷贝文件级属性
            for attr_key, attr_value in src.attrs.items():
                dst.attrs[attr_key] = attr_value
            
            # 处理每个数据集/分组
            for key in src.keys():
                if key == prompt_field:
                    dataset = src[prompt_field]
                    original_data = dataset[:]
                    original_shape = original_data.shape
                    
                    encoded_prompt = new_prompt.encode('utf-8')
                    max_length = max(len(encoded_prompt), dataset.dtype.itemsize)
                    new_dtype = f'|S{max_length}'
                    new_data = np.full(original_shape, encoded_prompt, dtype=new_dtype)
                    
                    new_dataset = dst.create_dataset(prompt_field, data=new_data)
                    # 拷贝原数据集的属性
                    for attr_key, attr_value in dataset.attrs.items():
                        new_dataset.attrs[attr_key] = attr_value
                else:
                    src.copy(key, dst)
        
        print(f"成功创建新文件 {output_file}")
        print(f"Prompt已从 '{info['current_prompt']}' 修改为 '{new_prompt}'")
        return True
        
    except Exception as e:
        print(f"修改文件时出错: {e}")
        return False


def batch_modify_prompts(directory: str, new_prompt: str, 
                        file_pattern: str = "*.hdf5",
                        prompt_field: str = 'task',
                        suffix: str = "_modified") -> dict:
    """
    批量修改目录中的HDF5文件prompt，并为每个文件生成新的副本
    
    Args:
        directory: 目录路径
        new_prompt: 新的prompt内容
        file_pattern: 文件模式（默认为"*.hdf5"）
        prompt_field: prompt字段名称
        suffix: 输出文件名后缀
    
    Returns:
        修改结果统计
    """
    import glob
    
    pattern = os.path.join(directory, file_pattern)
    files = glob.glob(pattern)
    
    results = {
        'success': [],
        'failed': [],
        'total': len(files)
    }
    
    print(f"找到 {len(files)} 个HDF5文件")
    
    for file_path in files:
        print(f"\n处理文件: {file_path}")
        # 批量模式下，generate_output_path会在modify_prompt_in_hdf5内部被调用
        # 传入None让函数自动生成路径
        if modify_prompt_in_hdf5(file_path, new_prompt, prompt_field, None):
            results['success'].append(file_path)
        else:
            results['failed'].append(file_path)
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="修改HDF5文件中的prompt内容",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 修改单个文件
  python modify_hdf5_prompt.py -f episode_001.hdf5 -p "抓取红色方块"
  
  # 批量修改目录中的所有HDF5文件
  python modify_hdf5_prompt.py -d ./recorded_data/blue -p "抓取蓝色方块"
  
  # 查看文件信息（不修改）
  python modify_hdf5_prompt.py -f episode_001.hdf5 --info
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', 
                      help='要修改的HDF5文件路径')
    group.add_argument('-d', '--directory', 
                      help='包含HDF5文件的目录路径')
    
    parser.add_argument('-p', '--prompt', 
                       help='新的prompt内容')
    parser.add_argument('--field', default='task',
                       help='prompt字段名称（默认：task）')
    parser.add_argument('-o', '--output',
                       help='单个文件模式下的输出文件路径')
    parser.add_argument('--suffix', default='_modified',
                       help='批量模式生成新文件时使用的后缀（默认：_modified）')
    parser.add_argument('--info', action='store_true',
                       help='仅显示文件信息，不进行修改')
    
    args = parser.parse_args()
    
    # 仅显示信息模式
    if args.info:
        if args.file:
            print(f"文件信息: {args.file}")
            info = check_hdf5_structure(args.file)
            print(f"包含task字段: {info.get('has_task', False)}")
            if info.get('has_task', False):
                print(f"数据形状: {info['task_shape']}")
                print(f"数据类型: {info['task_dtype']}")
                print(f"当前prompt: {info['current_prompt']}")
                print(f"唯一prompts: {info['unique_prompts']}")
            else:
                print(f"可用字段: {info.get('available_fields', [])}")
        return
    
    # 检查是否提供了prompt
    if not args.prompt:
        print("错误：必须提供新的prompt内容（-p/--prompt）")
        return
    
    # 单文件修改
    if args.file:
        success = modify_prompt_in_hdf5(
            args.file, 
            args.prompt, 
            args.field,
            args.output
        )
        if not success:
            exit(1)
    
    # 批量修改
    elif args.directory:
        results = batch_modify_prompts(
            args.directory, 
            args.prompt,
            prompt_field=args.field,
            suffix=args.suffix
        )
        
        print(f"\n修改完成:")
        print(f"成功: {len(results['success'])} 个文件")
        print(f"失败: {len(results['failed'])} 个文件")
        print(f"总计: {results['total']} 个文件")
        
        if results['failed']:
            print("\n失败的文件:")
            for file_path in results['failed']:
                print(f"  - {file_path}")
            exit(1)


if __name__ == "__main__":
    main()