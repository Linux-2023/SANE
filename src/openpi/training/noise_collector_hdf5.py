import h5py
import logging
import threading
import queue
from pathlib import Path
from typing import Any, Dict
from collections import defaultdict
import numpy as np

logger = logging.getLogger("openpi")

class NoiseCollectorHDF5:
    """
    收集训练噪声数据并以HDF5格式组织保存。
    
    对于每个(episode_index, frame_index)对，收集训练过程中遇到的所有噪声样本，
    然后以HDF5格式保存，支持压缩以节省空间。
    
    相比JSONL格式，HDF5可以节省10-16倍的存储空间。
    """
    
    def __init__(
        self, 
        output_dir: str | Path, 
        compression: str = "gzip",
        compression_opts: int = 5,
        save_interval: int = 1000
    ):
        """
        初始化HDF5噪声收集器。
        
        Args:
            output_dir: 输出目录
            compression: 压缩算法 ("gzip", "lzf", None)
            compression_opts: 压缩级别 (gzip: 0-9, lzf无此参数)
            save_interval: 每收集多少个batch后保存一次
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "noise_data.h5"
        self.save_interval = save_interval
        self.compression = compression
        self.compression_opts = compression_opts if compression == "gzip" else None
        
        # 缓冲区: {(episode_idx, frame_idx): [samples]}
        # 每个sample是一个numpy数组: [noise (1600), action (1600), time (1), episode_idx (1), frame_idx (1)]
        self.data_buffer = defaultdict(list)
        self.collection_count = 0
        
        # 线程安全队列用于异步处理
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        
        logger.info(f"NoiseCollectorHDF5 started, will save to {self.output_file}")
        logger.info(f"Compression: {compression}, level: {compression_opts}")

    def collect(self, batch_data: Dict[str, Any]):
        """
        收集一个batch的训练数据。
        
        期望的batch_data键：
        - 'episode_index': 数组，shape (batch_size,)
        - 'frame_index': 数组，shape (batch_size,)
        - 'noise': 数组，shape (batch_size, action_horizon, action_dim)
        - 'time': 数组，shape (batch_size,)
        - 'actions': 数组，shape (batch_size, action_horizon, action_dim)
        """
        # 放入队列进行异步处理
        self.queue.put(batch_data)

    def _writer_loop(self):
        """后台线程处理收集的数据。"""
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                batch_data = self.queue.get(timeout=1.0)
                self._process_batch(batch_data)
                
                # 定期保存到磁盘
                self.collection_count += 1
                if self.collection_count % self.save_interval == 0:
                    self._save_to_hdf5()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in NoiseCollectorHDF5 writer loop: {e}", exc_info=True)

    def _process_batch(self, batch_data: Dict[str, Any]):
        """处理batch并添加到缓冲区。"""
        episode_indices = batch_data["episode_index"]
        frame_indices = batch_data["frame_index"]
        noises = batch_data["noise"]
        times = batch_data["time"]
        actions = batch_data["actions"]
        
        batch_size = len(episode_indices)
        
        for i in range(batch_size):
            episode_idx = int(episode_indices[i])
            frame_idx = int(frame_indices[i])
            key = (episode_idx, frame_idx)
            
            # 将noise和action展平为1D数组
            noise_flat = np.array(noises[i]).flatten()  # shape: (action_horizon * action_dim,)
            action_flat = np.array(actions[i]).flatten()  # shape: (action_horizon * action_dim,)
            time_val = float(times[i])
            
            # 组合成单个样本数组: [noise, action, time, episode_idx, frame_idx]
            sample = np.concatenate([
                noise_flat,
                action_flat,
                [time_val, episode_idx, frame_idx]
            ])
            
            self.data_buffer[key].append(sample)

    def _save_to_hdf5(self):
        """将缓冲区数据保存到HDF5文件（增量保存）。"""
        if not self.data_buffer:
            return
            
        try:
            # 使用'a'模式以支持增量写入
            with h5py.File(self.output_file, 'a') as f:
                for (episode_idx, frame_idx), samples in self.data_buffer.items():
                    # 创建分层路径
                    episode_group_name = f"episode_{episode_idx}"
                    frame_dataset_name = f"{episode_group_name}/frame_{frame_idx}/samples"
                    
                    # 将样本列表转换为numpy数组
                    samples_array = np.array(samples)  # shape: (num_samples, feature_dim)
                    
                    # 检查数据集是否已存在
                    if frame_dataset_name in f:
                        # 已存在，需要追加数据
                        existing_data = f[frame_dataset_name][:]
                        combined_data = np.vstack([existing_data, samples_array])
                        
                        # 删除旧数据集并创建新的
                        del f[frame_dataset_name]
                        f.create_dataset(
                            frame_dataset_name,
                            data=combined_data,
                            compression=self.compression,
                            compression_opts=self.compression_opts
                        )
                    else:
                        # 不存在，直接创建
                        f.create_dataset(
                            frame_dataset_name,
                            data=samples_array,
                            compression=self.compression,
                            compression_opts=self.compression_opts
                        )
                    
                    # 添加元数据属性
                    f[frame_dataset_name].attrs['episode_index'] = episode_idx
                    f[frame_dataset_name].attrs['frame_index'] = frame_idx
                    f[frame_dataset_name].attrs['num_samples'] = len(samples_array)
            
            logger.info(f"Saved {len(self.data_buffer)} (episode, frame) pairs to {self.output_file}")
            
            # 清空缓冲区（已保存）
            self.data_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to save to HDF5: {e}", exc_info=True)

    def close(self):
        """停止收集并保存所有数据。"""
        logger.info("NoiseCollectorHDF5 closing, waiting for queue to empty...")
        self.stop_event.set()
        self.writer_thread.join()
        
        # 保存剩余数据
        logger.info("Saving final data...")
        self._save_to_hdf5()
        
        # 生成摘要统计
        self._save_summary()
        
        logger.info("NoiseCollectorHDF5 stopped")

    def _save_summary(self):
        """保存摘要统计信息。"""
        try:
            with h5py.File(self.output_file, 'r') as f:
                episodes = set()
                total_frames = 0
                total_samples = 0
                
                def count_samples(name, obj):
                    nonlocal total_frames, total_samples
                    if isinstance(obj, h5py.Dataset) and 'samples' in name:
                        total_frames += 1
                        total_samples += obj.attrs.get('num_samples', len(obj))
                        episode_idx = obj.attrs.get('episode_index', -1)
                        episodes.add(episode_idx)
                
                f.visititems(count_samples)
                
                summary = {
                    'total_episodes': len(episodes),
                    'total_frames': total_frames,
                    'total_samples': total_samples,
                    'compression': self.compression,
                    'compression_opts': self.compression_opts,
                }
                
                # 保存为文本文件
                summary_file = self.output_dir / "noise_collection_summary.txt"
                with open(summary_file, 'w') as sf:
                    sf.write("=== Noise Data Collection Summary ===\n")
                    for key, value in summary.items():
                        sf.write(f"{key}: {value}\n")
                    
                    # 文件大小
                    file_size_mb = self.output_file.stat().st_size / (1024 * 1024)
                    sf.write(f"file_size_mb: {file_size_mb:.2f}\n")
                
                logger.info(f"Summary: {summary}, file size: {file_size_mb:.2f} MB")
                
        except Exception as e:
            logger.error(f"Failed to save summary: {e}", exc_info=True)
