#!/usr/bin/env python3
"""
简洁的工具脚本，用于读取 HDF5 文件并打印所有键

用法:
    python inspect_hdf5.py <hdf5_file_path>
"""

import sys
from pathlib import Path
import h5py


def print_hdf5_keys(file_path: Path):
    """递归打印 HDF5 文件中的所有键"""
    with h5py.File(file_path, "r") as f:
        def print_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name} (shape={obj.shape}, dtype={obj.dtype})")
            else:
                print(f"{name}/")
        
        print(f"File: {file_path}\n")
        f.visititems(print_keys)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_hdf5.py <hdf5_file_path>")
        sys.exit(1)
    
    hdf5_path = Path(sys.argv[1])
    if not hdf5_path.exists():
        print(f"Error: File not found: {hdf5_path}")
        sys.exit(1)
    
    print_hdf5_keys(hdf5_path)

