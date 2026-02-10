#!/usr/bin/env python3
"""
为目录下的所有 HDF5 文件新增字符串字段 advantage，值为 "advantage: positive"。
"""

import argparse
import glob
import os
from typing import Optional

import h5py
import numpy as np


def infer_length(f: h5py.File) -> Optional[int]:
    """
    推断步数，优先使用与 prompt 同步的 task 数据集，其次使用 action/observations 形状。
    """
    if "task" in f:
        return len(f["task"])
    if "action" in f:
        return f["action"].shape[0]
    if "observations" in f and "qpos" in f["observations"]:
        return f["observations"]["qpos"].shape[0]
    return None


def add_advantage(path: str, dataset_name: str, value: str) -> bool:
    """
    为单个 HDF5 文件添加 advantage 数据集。
    """
    if not os.path.exists(path):
        print(f"[跳过] 文件不存在: {path}")
        return False

    try:
        with h5py.File(path, "r+") as f:
            if dataset_name in f:
                print(f"[已存在] {dataset_name} 于 {path}")
                return True

            num_steps = infer_length(f) or 1
            encoded = value.encode("utf-8")
            max_len = max(len(encoded), 16)
            data = np.full((num_steps,), encoded, dtype=f"|S{max_len}")

            dset = f.create_dataset(dataset_name, data=data)
            dset.attrs["description"] = "auto-added advantage string"

        print(f"[完成] 已为 {path} 添加 {dataset_name}，长度 {num_steps}")
        return True
    except Exception as e:
        print(f"[失败] 处理 {path} 出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="批量为 HDF5 添加 advantage 字段")
    parser.add_argument("-d", "--directory", required=True, help="包含 HDF5 的目录")
    parser.add_argument(
        "-p", "--pattern", default="*.hdf5", help="文件匹配模式（默认 *.hdf5）"
    )
    parser.add_argument(
        "-n", "--name", default="advantage", help="新增数据集名称（默认 advantage）"
    )
    parser.add_argument(
        "-v",
        "--value",
        default="advantage: positive",
        help='写入的字符串值（默认 "advantage: positive"）',
    )
    args = parser.parse_args()

    pattern = os.path.join(args.directory, args.pattern)
    files = glob.glob(pattern)
    print(f"找到 {len(files)} 个文件匹配 {pattern}")

    success, failed = [], []
    for file_path in files:
        if add_advantage(file_path, args.name, args.value):
            success.append(file_path)
        else:
            failed.append(file_path)

    print(f"\n完成: 成功 {len(success)} 个，失败 {len(failed)} 个，共 {len(files)} 个")
    if failed:
        print("失败列表:")
        for fpath in failed:
            print(f"  - {fpath}")


if __name__ == "__main__":
    main()

