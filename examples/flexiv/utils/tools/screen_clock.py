#!/usr/bin/env python3
"""
超高精度屏幕时钟
用于摄像头同步测试 - 显示微秒级时间戳
"""

import cv2
import numpy as np
import time
from datetime import datetime
import argparse


def create_clock_display(width=1920, height=1080):
    """创建高精度时钟显示"""
    # 深色背景
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (15, 15, 20)
    
    # 获取当前时间（微秒精度）
    now = time.time()
    now_ns = time.time_ns()  # 纳秒精度
    us = (now_ns // 1000) % 1000000  # 微秒
    ms = us // 1000
    us_only = us % 1000
    dt = datetime.fromtimestamp(now)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # ===== 主时间显示区域 =====
    # HH:MM:SS
    time_str = dt.strftime("%H:%M:%S")
    font_scale = min(width, height) / 300
    thickness = max(2, int(font_scale * 2))
    
    (tw, th), _ = cv2.getTextSize(time_str, font, font_scale, thickness)
    x_time = (width - tw) // 2 - int(width * 0.15)
    y_time = int(height * 0.35)
    
    # 时间阴影效果
    cv2.putText(img, time_str, (x_time + 3, y_time + 3), font, font_scale, (30, 30, 30), thickness + 2)
    cv2.putText(img, time_str, (x_time, y_time), font, font_scale, (255, 255, 255), thickness)
    
    # .毫秒 (黄色)
    ms_str = f".{ms:03d}"
    font_scale_ms = font_scale * 0.8
    x_ms = x_time + tw + 10
    cv2.putText(img, ms_str, (x_ms, y_time), font, font_scale_ms, (0, 220, 255), max(1, thickness - 1))
    
    # 微秒 (青色，更小)
    us_str = f"{us_only:03d}"
    font_scale_us = font_scale * 0.5
    (usw, _), _ = cv2.getTextSize(ms_str, font, font_scale_ms, thickness)
    x_us = x_ms + usw + 5
    cv2.putText(img, us_str, (x_us, y_time), font, font_scale_us, (0, 255, 200), max(1, thickness - 2))
    cv2.putText(img, "us", (x_us + int(font_scale_us * 50), y_time), font, font_scale_us * 0.6, (100, 200, 150), 1)
    
    # ===== Unix 时间戳（高精度）=====
    unix_str = f"{now:.6f}"
    y_unix = int(height * 0.50)
    cv2.putText(img, "UNIX:", (int(width * 0.25), y_unix), font, font_scale * 0.4, (100, 150, 100), max(1, thickness - 2))
    cv2.putText(img, unix_str, (int(width * 0.35), y_unix), font, font_scale * 0.5, (120, 255, 120), max(1, thickness - 1))
    
    # ===== 视觉同步指示器 =====
    
    # 1. 毫秒进度条 (每秒循环一次)
    bar_y = int(height * 0.58)
    bar_h = 30
    bar_x1 = int(width * 0.1)
    bar_x2 = int(width * 0.9)
    bar_w = bar_x2 - bar_x1
    
    # 背景条
    cv2.rectangle(img, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(img, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (80, 80, 80), 1)
    
    # 进度 (毫秒)
    progress = ms / 1000.0
    progress_x = bar_x1 + int(bar_w * progress)
    
    # 渐变色进度条
    color_r = int(255 * progress)
    color_g = int(255 * (1 - abs(progress - 0.5) * 2))
    color_b = int(255 * (1 - progress))
    cv2.rectangle(img, (bar_x1, bar_y), (progress_x, bar_y + bar_h), (color_b, color_g, color_r), -1)
    
    # 当前位置指示器
    cv2.line(img, (progress_x, bar_y - 5), (progress_x, bar_y + bar_h + 5), (255, 255, 255), 2)
    
    # 毫秒刻度 (每100ms)
    for i in range(11):
        tick_x = bar_x1 + int(bar_w * i / 10)
        cv2.line(img, (tick_x, bar_y + bar_h), (tick_x, bar_y + bar_h + 8), (150, 150, 150), 1)
        if i % 5 == 0:
            cv2.putText(img, f"{i*100}", (tick_x - 15, bar_y + bar_h + 25), font, 0.4, (150, 150, 150), 1)
    
    cv2.putText(img, "ms", (bar_x2 + 10, bar_y + 20), font, 0.5, (150, 150, 150), 1)
    
    # 2. 颜色闪烁区域 (10Hz, 每100ms变色)
    flash_y = int(height * 0.70)
    flash_h = 60
    color_idx = int(now * 10) % 10
    colors = [
        (66, 66, 255),    # 红
        (66, 165, 255),   # 橙
        (66, 255, 255),   # 黄
        (66, 255, 66),    # 黄绿
        (255, 255, 66),   # 青
        (255, 165, 66),   # 蓝绿
        (255, 66, 66),    # 蓝
        (255, 66, 165),   # 紫
        (165, 66, 255),   # 粉
        (255, 255, 255),  # 白
    ]
    bar_color = colors[color_idx]
    cv2.rectangle(img, (bar_x1, flash_y), (bar_x2, flash_y + flash_h), bar_color, -1)
    
    # 颜色索引标签
    cv2.putText(img, f"100ms Block: {color_idx}", (bar_x1, flash_y - 10), font, 0.6, (200, 200, 200), 1)
    
    # 3. 高频闪烁方块 (左右两个，交替闪烁)
    sq_size = 80
    sq_y = int(height * 0.82)
    
    # 左方块: 30Hz (每33ms)
    left_state = int(now * 30) % 2
    left_color = (255, 255, 255) if left_state else (30, 30, 30)
    cv2.rectangle(img, (bar_x1, sq_y), (bar_x1 + sq_size, sq_y + sq_size), left_color, -1)
    cv2.putText(img, "30Hz", (bar_x1, sq_y + sq_size + 20), font, 0.5, (150, 150, 150), 1)
    
    # 中间方块: 60Hz (每16.6ms)
    mid_x = width // 2 - sq_size // 2
    mid_state = int(now * 60) % 2
    mid_color = (255, 255, 255) if mid_state else (30, 30, 30)
    cv2.rectangle(img, (mid_x, sq_y), (mid_x + sq_size, sq_y + sq_size), mid_color, -1)
    cv2.putText(img, "60Hz", (mid_x, sq_y + sq_size + 20), font, 0.5, (150, 150, 150), 1)
    
    # 右方块: 120Hz (每8.3ms)
    right_state = int(now * 120) % 2
    right_color = (255, 255, 255) if right_state else (30, 30, 30)
    cv2.rectangle(img, (bar_x2 - sq_size, sq_y), (bar_x2, sq_y + sq_size), right_color, -1)
    cv2.putText(img, "120Hz", (bar_x2 - sq_size, sq_y + sq_size + 20), font, 0.5, (150, 150, 150), 1)
    
    # 4. 微秒级旋转指针 (模拟表盘)
    dial_x = int(width * 0.85)
    dial_y = int(height * 0.30)
    dial_r = min(width, height) // 10
    
    # 表盘背景
    cv2.circle(img, (dial_x, dial_y), dial_r, (50, 50, 50), -1)
    cv2.circle(img, (dial_x, dial_y), dial_r, (100, 100, 100), 2)
    
    # 刻度
    for i in range(12):
        angle = i * 30 * np.pi / 180
        x1 = int(dial_x + (dial_r - 8) * np.sin(angle))
        y1 = int(dial_y - (dial_r - 8) * np.cos(angle))
        x2 = int(dial_x + dial_r * np.sin(angle))
        y2 = int(dial_y - dial_r * np.cos(angle))
        cv2.line(img, (x1, y1), (x2, y2), (150, 150, 150), 2)
    
    # 毫秒指针 (每秒转一圈)
    ms_angle = (ms / 1000.0) * 2 * np.pi
    ms_x = int(dial_x + (dial_r - 15) * np.sin(ms_angle))
    ms_y = int(dial_y - (dial_r - 15) * np.cos(ms_angle))
    cv2.line(img, (dial_x, dial_y), (ms_x, ms_y), (0, 200, 255), 3)
    
    # 微秒指针 (每毫秒转一圈，更快)
    us_angle = (us_only / 1000.0) * 2 * np.pi
    us_x = int(dial_x + (dial_r - 25) * np.sin(us_angle))
    us_y = int(dial_y - (dial_r - 25) * np.cos(us_angle))
    cv2.line(img, (dial_x, dial_y), (us_x, us_y), (0, 255, 200), 2)
    
    # 中心点
    cv2.circle(img, (dial_x, dial_y), 5, (255, 255, 255), -1)
    
    cv2.putText(img, "ms/us dial", (dial_x - 40, dial_y + dial_r + 25), font, 0.4, (150, 150, 150), 1)
    
    # 5. 二进制时间编码 (用于自动化分析)
    bin_y = int(height * 0.95)
    bit_w = 20
    bit_h = 15
    bin_x = int(width * 0.1)
    
    # 编码当前毫秒 (10位二进制)
    for i in range(10):
        bit = (ms >> (9 - i)) & 1
        color = (255, 255, 255) if bit else (30, 30, 30)
        x = bin_x + i * (bit_w + 3)
        cv2.rectangle(img, (x, bin_y), (x + bit_w, bin_y + bit_h), color, -1)
    
    cv2.putText(img, f"Binary ms: {ms:010b}", (bin_x, bin_y - 8), font, 0.4, (100, 100, 100), 1)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="超高精度屏幕时钟")
    parser.add_argument("--width", type=int, default=1920, help="窗口宽度")
    parser.add_argument("--height", type=int, default=1080, help="窗口高度")
    parser.add_argument("--fullscreen", "-f", action="store_true", help="全屏显示")
    parser.add_argument("--monitor", type=int, default=0, help="显示器编号")
    args = parser.parse_args()
    
    window_name = "Ultra High Precision Clock"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, args.width, args.height)
    
    print("=" * 60)
    print("  超高精度屏幕时钟 (微秒级)")
    print("=" * 60)
    print("  显示元素:")
    print("    - 时间: HH:MM:SS.mmm uuu (毫秒+微秒)")
    print("    - Unix时间戳: 6位小数精度")
    print("    - 毫秒进度条: 0-1000ms 可视化")
    print("    - 100ms色块: 10种颜色循环")
    print("    - 闪烁方块: 30Hz / 60Hz / 120Hz")
    print("    - 表盘: 毫秒指针(黄) + 微秒指针(青)")
    print("    - 二进制编码: 10位毫秒编码")
    print("=" * 60)
    print("  按键: 'q' 退出, 'f' 切换全屏")
    print("=" * 60)
    
    fullscreen = args.fullscreen
    frame_count = 0
    start_time = time.time()
    
    while True:
        img = create_clock_display(args.width, args.height)
        
        # 显示刷新率
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
            cv2.putText(img, f"Display FPS: {fps:.1f}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        cv2.imshow(window_name, img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
