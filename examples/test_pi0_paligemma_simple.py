#!/usr/bin/env python3
"""
简化版本：最小化的PI0模型测试代码。

演示如何：
1. 加载PI0模型
2. 加载LeRobot数据集
3. 使用PaliGemma处理批次数据
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['HF_HOME'] = "/workspace/pjk/ELM/openpi/huggingface"

import torch
from pathlib import Path
import time

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from transformers import AutoProcessor


def main():
    # 配置参数
    CONFIG_NAME = "pi05_flexiv_pose_train"
    CHECKPOINT_DIR = "checkpoints/openpi/openpi-assets/checkpoints/pi05_base_pytorch"
    DATASET_PATH = "/workspace/pjk/ELM/openpi/datasets/flexiv/gripperpap/flexiv_lerobot_data_pose"
    BATCH_SIZE = 1
    NUM_BATCHES = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {DEVICE}")
    print("\n" + "="*60)
    print("开始加载资源...")
    print("="*60)
    
    # 1. 加载模型
    print("\n[1/3] 正在加载模型（可能需要1-3分钟）...")
    t0 = time.time()
    config = _config.get_config(CONFIG_NAME)
    
    # 禁用梯度计算以加快加载速度
    with torch.no_grad():
        policy = _policy_config.create_trained_policy(config, CHECKPOINT_DIR, pytorch_device=DEVICE)
    
    model = policy._model
    paligemma = model.paligemma_with_expert.paligemma
    paligemma.eval()
    
    # 将模型转换为float32以避免dtype不匹配
    paligemma = paligemma.to(torch.float32)
    
    # 禁用梯度以加快推理速度
    for param in paligemma.parameters():
        param.requires_grad = False
    
    model_time = time.time() - t0
    print(f"✓ 模型加载完成 (耗时: {model_time:.1f}秒)")
    
    # 2. 加载数据集
    print(f"\n[2/3] 正在加载数据集...")
    t1 = time.time()
    dataset = lerobot_dataset.LeRobotDataset(repo_id="flexiv_lerobot_data_pose", root=DATASET_PATH)
    dataset_time = time.time() - t1
    print(f"✓ 数据集加载完成 (总样本数: {len(dataset)}, 耗时: {dataset_time:.1f}秒)")

    # 3. 加载 PaliGemma 处理器用于生成文本
    print(f"\n[3/3] 正在加载文本处理器...")
    t2 = time.time()
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
    processor_time = time.time() - t2
    print(f"✓ 处理器加载完成 (耗时: {processor_time:.1f}秒)")
    
    total_load_time = model_time + dataset_time + processor_time
    print(f"\n总加载时间: {total_load_time:.1f}秒")

    from PIL import Image
    import requests
    prompt = "<image> What is the animals in the picture?"
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    
    # 将inputs迁移到GPU，pixel_values转为float32，其他张量保持原始dtype
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            if key == 'pixel_values':
                inputs[key] = inputs[key].to(DEVICE).to(torch.float32)
            else:
                inputs[key] = inputs[key].to(DEVICE)

    # Generate
    generate_ids = paligemma.generate(**inputs)
    answer = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("Answer:", answer)
    
    # generation_kwargs = {
    #     "max_new_tokens": 32,
    #     "do_sample": False,
    # }

    # # 4. 创建数据加载器并处理批次
    # print("\n" + "="*60)
    # print(f"处理{NUM_BATCHES}个批次...")
    # print("="*60)
    
    # from torch.utils.data import DataLoader
    
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     drop_last=True,
    # )
    
    # for batch_idx, batch in enumerate(dataloader):
    #     if batch_idx >= NUM_BATCHES:
    #         break
        
    #     print(f"\n--- Batch {batch_idx + 1}/{NUM_BATCHES} ---")
        
    #     # LeRobot数据集返回字典格式的批次数据
    #     # 每个样本包含: observation.state, action, observation.images.cam_left_wrist, observation.images.cam_right_wrist等
    #     if not isinstance(batch, dict):
    #         print(f"Warning: Expected dict batch, got {type(batch)}")
    #         continue
        
    #     # 获取图像 - 使用cam_left_wrist或cam_right_wrist
    #     # batch中的图像已经是批处理后的形状: [batch_size, 3, 224, 224]
    #     if 'observation.images.cam_left_wrist' in batch:
    #         images = batch['observation.images.cam_left_wrist']
    #         camera_name = "cam_left_wrist"
    #     elif 'observation.images.cam_right_wrist' in batch:
    #         images = batch['observation.images.cam_right_wrist']
    #         camera_name = "cam_right_wrist"
    #     else:
    #         print("❌ No camera images found in batch. Available keys:", list(batch.keys()))
    #         continue
        
    #     # 确保是torch.Tensor并移到正确的设备
    #     if not isinstance(images, torch.Tensor):
    #         images = torch.tensor(images)
        
    #     images = images.to(DEVICE)
        
    #     print(f"Image shape ({camera_name}): {images.shape}")
        
    #     # 使用PaliGemma处理图像并生成回答
    #     with torch.inference_mode():
    #         # 方式1：获取图像特征
    #         image_features = paligemma.model.get_image_features(images)
    #         print(f"Image features shape: {image_features.shape}")

    #         # 方式2：使用 generate 生成文本描述
    #         try:
    #             # PaliGemma的正确prompt格式不需要手动添加<image>，processor会自动处理
    #             prompt = "<image> What objects do you see in this image?"
    #             image_list = [img.cpu() for img in images]
    #             inputs = processor(images=image_list, text=[prompt] * images.shape[0], return_tensors="pt", padding=True)
                
    #             # 只将图像数据(pixel_values)转换为float32，input_ids必须保持Long类型
    #             for key in inputs:
    #                 if torch.is_tensor(inputs[key]):
    #                     if key == 'pixel_values':
    #                         inputs[key] = inputs[key].to(DEVICE).to(torch.float32)
    #                     else:
    #                         inputs[key] = inputs[key].to(DEVICE)

    #             generate_ids = paligemma.generate(**inputs, **generation_kwargs)
    #             generated_text = processor.batch_decode(
    #                 generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    #             )

    #             for i, text in enumerate(generated_text):
    #                 print(f"  Answer[{i}]: {text}")
    #         except Exception as e:
    #             print(f"  Note: 文本生成失败 ({type(e).__name__}: {str(e)[:60]}...)")
            
    #         print(f"✓ Batch {batch_idx + 1} 处理完成")
    
    # print("\n✓ 所有批次处理完成!")


if __name__ == "__main__":
    main()
