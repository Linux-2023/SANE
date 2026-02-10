import json
import logging
import threading
import queue
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict
import numpy as np
import jax

logger = logging.getLogger("openpi")

class NoiseCollector:
    """
    Collects training noise data and organizes it by episode.
    
    For each (episode_index, frame_index) pair, collects all noise samples 
    encountered during training, then saves the data organized by episode.
    """
    
    def __init__(self, output_dir: str | Path, save_interval: int = 1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "noise_data_by_episode.jsonl"
        self.save_interval = save_interval
        
        # Store data as: {(episode_idx, frame_idx): [{"noise": ..., "time": ..., "action": ...}, ...]}
        self.data_buffer = defaultdict(list)
        self.collection_count = 0
        
        # Thread-safe queue for async processing
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        
        logger.info(f"NoiseCollector started, will save to {self.output_file}")

    def collect(self, batch_data: Dict[str, Any]):
        """
        Collect a batch of training data.
        
        Expected keys in batch_data:
        - 'episode_index': array of shape (batch_size,)
        - 'frame_index': array of shape (batch_size,)
        - 'noise': array of shape (batch_size, action_horizon, action_dim)
        - 'time': array of shape (batch_size,)
        - 'actions': array of shape (batch_size, action_horizon, action_dim)
        """
        # Put into queue for async processing
        self.queue.put(batch_data)

    def _writer_loop(self):
        """Background thread that processes collected data."""
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                batch_data = self.queue.get(timeout=1.0)
                self._process_batch(batch_data)
                
                # Periodically save to disk
                self.collection_count += 1
                if self.collection_count % self.save_interval == 0:
                    self._save_incremental()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in NoiseCollector writer loop: {e}", exc_info=True)

    def _process_batch(self, batch_data: Dict[str, Any]):
        """Process a batch and add individual samples to buffer."""
        episode_indices = batch_data["episode_index"]
        frame_indices = batch_data["frame_index"]
        noises = batch_data["noise"]
        times = batch_data["time"]
        actions = batch_data["actions"]
        
        batch_size = len(episode_indices)
        
        for i in range(batch_size):
            key = (int(episode_indices[i]), int(frame_indices[i]))
            sample = {
                "noise": noises[i].tolist() if hasattr(noises[i], "tolist") else noises[i],
                "time": float(times[i]),
                "action": actions[i].tolist() if hasattr(actions[i], "tolist") else actions[i],
            }
            self.data_buffer[key].append(sample)

    def _save_incremental(self):
        """Save current buffer state (incremental backup)."""
        try:
            backup_file = self.output_dir / f"noise_data_backup_{self.collection_count}.json"
            with open(backup_file, "w") as f:
                # Convert buffer to serializable format
                serializable = {f"{k[0]}_{k[1]}": v for k, v in self.data_buffer.items()}
                json.dump(serializable, f, indent=2)
            logger.info(f"Saved incremental backup to {backup_file}")
        except Exception as e:
            logger.error(f"Failed to save incremental backup: {e}")

    def _organize_by_episode(self) -> Dict[int, List[Dict]]:
        """Organize collected data by episode."""
        episodes = defaultdict(list)
        
        for (episode_idx, frame_idx), samples in self.data_buffer.items():
            episodes[episode_idx].append({
                "frame_index": frame_idx,
                "samples": samples,
            })
        
        # Sort frames within each episode
        for episode_idx in episodes:
            episodes[episode_idx].sort(key=lambda x: x["frame_index"])
        
        return episodes

    def save_final(self):
        """Save final organized data by episode."""
        try:
            episodes = self._organize_by_episode()
            
            # Save as JSONL, one episode per line
            with open(self.output_file, "w") as f:
                for episode_idx in sorted(episodes.keys()):
                    episode_data = {
                        "episode_index": episode_idx,
                        "frames": episodes[episode_idx],
                    }
                    f.write(json.dumps(episode_data) + "\n")
            
            logger.info(f"Saved final organized data to {self.output_file}")
            logger.info(f"Total episodes: {len(episodes)}, total samples: {len(self.data_buffer)}")
            
            # Also save summary statistics
            summary_file = self.output_dir / "noise_collection_summary.json"
            summary = {
                "total_episodes": len(episodes),
                "total_frame_samples": len(self.data_buffer),
                "episodes": {
                    episode_idx: {
                        "num_frames": len(frames),
                        "total_noise_samples": sum(len(f["samples"]) for f in frames),
                    }
                    for episode_idx, frames in episodes.items()
                }
            }
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save final data: {e}", exc_info=True)

    def close(self):
        """Stop collection and save all data."""
        logger.info("NoiseCollector closing, waiting for queue to empty...")
        self.stop_event.set()
        self.writer_thread.join()
        
        logger.info("Saving final organized data...")
        self.save_final()
        logger.info("NoiseCollector stopped")
