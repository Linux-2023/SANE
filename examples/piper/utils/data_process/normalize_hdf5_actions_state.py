#!/usr/bin/env python3
"""
批量归一化 HDF5 文件中的 state 与 action 的最后一维。

脚本会扫描给定目录下的所有 .hdf5 文件，为每个文件创建一个新的副本，
并将指定数据集（默认：`/observations/qpos` 与 `/action`）的最后一维
线性映射到 [0.0, 0.07] 区间。

使用示例：
    python normalize_hdf5_actions_state.py --directory ./recorded_data
    python normalize_hdf5_actions_state.py --directory ./data --inplace
    python normalize_hdf5_actions_state.py --directory ./data \
        --state-path observations/qpos --action-path action --suffix _norm
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
from typing import Optional

import h5py
import numpy as np


TARGET_MIN = 0.0
TARGET_MAX = 0.07

# 默认数据集路径
DEFAULT_STATE_PATH = "observations/qpos"
DEFAULT_ACTION_PATH = "action"


def normalize_last_axis(data: np.ndarray) -> np.ndarray:
    """将数组最后一维度线性映射至 [TARGET_MIN, TARGET_MAX]。"""
    print(f"normalize_last_axis: {data.shape}")

    data[..., -1] = (data[..., -1] - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)


    return data


def process_dataset(
    file_handle: h5py.File,
    dataset_path: str,
) -> bool:
    """对指定路径的数据集执行归一化。返回是否成功处理。"""
    if dataset_path not in file_handle:
        print(f"  ⚠️  数据集路径 '{dataset_path}' 不存在，跳过")
        return False

    dataset = file_handle[dataset_path]
    data = dataset[...]

    normalized = normalize_last_axis(data)
    dataset[...] = normalized.astype(dataset.dtype, copy=False)
    print(f"  ✅ 已归一化 '{dataset_path}'")
    return True


def process_file(
    file_path: str,
    state_path: str,
    action_path: str,
    inplace: bool,
    suffix: str,
) -> Optional[str]:
    """处理单个 HDF5 文件，返回输出文件路径。"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在：{file_path}")
        return None

    if inplace:
        target_path = file_path
    else:
        directory, filename = os.path.split(file_path)
        name, ext = os.path.splitext(filename)
        target_path = os.path.join(directory, f"{name}{suffix}{ext}")
        shutil.copy2(file_path, target_path)
        print(f"📄 已创建副本：{target_path}")

    with h5py.File(target_path, "r+") as f:
        print(f"\n=== 处理文件: {target_path} ===")
        state_done = process_dataset(f, state_path)
        action_done = process_dataset(f, action_path)
        if not state_done and not action_done:
            print("  ⚠️ 未处理任何数据集")

    return target_path


def main():
    parser = argparse.ArgumentParser(
        description="批量归一化 HDF5 文件中的 state 与 action",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="包含 HDF5 文件的目录",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.hdf5",
        help="匹配 HDF5 文件的通配符",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default=DEFAULT_STATE_PATH,
        help="state 数据集路径",
    )
    parser.add_argument(
        "--action-path",
        type=str,
        default=DEFAULT_ACTION_PATH,
        help="action 数据集路径",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="直接在原文件上修改（默认创建带后缀的新文件）",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_normalized",
        help="非 inplace 模式下，新文件名使用的后缀",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"❌ 目录不存在：{args.directory}")
        return

    pattern = os.path.join(args.directory, args.pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"⚠️ 未找到匹配的 HDF5 文件：{pattern}")
        return

    print(f"🔍 找到 {len(files)} 个 HDF5 文件")
    processed = 0
    outputs = []
    for file_path in files:
        result = process_file(
            file_path,
            state_path=args.state_path,
            action_path=args.action_path,
            inplace=args.inplace,
            suffix=args.suffix,
        )
        if result is not None:
            processed += 1
            outputs.append(result)

    print(f"\n✅ 完成，处理成功 {processed}/{len(files)} 个文件")
    if outputs:
        print("输出文件：")
        for path in outputs:
            print(f"  - {path}")


if __name__ == "__main__":
    main()

