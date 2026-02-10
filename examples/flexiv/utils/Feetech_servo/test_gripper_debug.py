#!/usr/bin/env python3
"""
夹爪调试测试脚本
用于诊断为什么 get_percent() 一直返回 0
"""

import sys
import json
import time
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))
from scservo_sdk import *
from gripper import Gripper


def test_gripper_connection(side: str = 'left'):
    """测试夹爪连接和读取"""
    print(f"\n{'='*60}")
    print(f"测试 {side} 夹爪")
    print(f"{'='*60}\n")
    
    # 1. 加载配置文件
    config_file = 'gripper_config.json'
    print(f"1. 加载配置文件: {config_file}")
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"   ✅ 配置文件加载成功")
        
        if side in config.get('grippers', {}):
            gripper_config = config['grippers'][side]
            print(f"   - 端口: {gripper_config.get('port', 'N/A')}")
            print(f"   - 舵机ID: {gripper_config.get('servo_id', 'N/A')}")
            print(f"   - 最小位置: {gripper_config.get('min_position', 'N/A')}")
            print(f"   - 最大位置: {gripper_config.get('max_position', 'N/A')}")
        else:
            print(f"   ❌ 配置文件中没有 {side} 夹爪的数据")
            return
    except FileNotFoundError:
        print(f"   ❌ 配置文件不存在: {config_file}")
        return
    except Exception as e:
        print(f"   ❌ 加载配置文件失败: {e}")
        return
    
    # 2. 创建夹爪实例
    print(f"\n2. 创建夹爪实例")
    try:
        gripper = Gripper(side=side, config_file=config_file)
        print(f"   ✅ 夹爪实例创建成功")
        print(f"   - 最大角度: {gripper.max_angle:.1f}°")
    except Exception as e:
        print(f"   ❌ 创建夹爪实例失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 连接夹爪
    print(f"\n3. 连接夹爪")
    try:
        if gripper.connect():
            print(f"   ✅ 连接成功")
        else:
            print(f"   ❌ 连接失败")
            return
    except Exception as e:
        print(f"   ❌ 连接异常: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 测试读取位置
    print(f"\n4. 测试读取电机位置（原始值）")
    try:
        # 直接读取位置
        port_handler = gripper._port_handler
        packet_handler = gripper._packet_handler
        servo_id = gripper._servo_id
        
        print(f"   - 舵机ID: {servo_id}")
        print(f"   - 尝试读取位置...")
        
        # 读取多次，看是否一致
        positions = []
        for i in range(5):
            pos, result, error = packet_handler.read2ByteTxRx(
                servo_id, SMS_STS_PRESENT_POSITION_L
            )
            if result == COMM_SUCCESS:
                positions.append(pos)
                print(f"   - 第 {i+1} 次读取: 位置 = {pos}, 结果 = COMM_SUCCESS")
            else:
                print(f"   - 第 {i+1} 次读取: 失败, 错误码 = {error}, 结果 = {result}")
            time.sleep(0.1)
        
        if positions:
            avg_pos = sum(positions) / len(positions)
            print(f"\n   ✅ 读取成功")
            print(f"   - 平均位置: {avg_pos:.1f}")
            print(f"   - 位置范围: {min(positions)} ~ {max(positions)}")
        else:
            print(f"\n   ❌ 所有读取都失败")
            return
            
    except Exception as e:
        print(f"   ❌ 读取位置异常: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 测试百分比转换
    print(f"\n5. 测试百分比转换")
    try:
        min_pos = gripper._min_position
        max_pos = gripper._max_position
        
        print(f"   - 最小位置 (min_position): {min_pos}")
        print(f"   - 最大位置 (max_position): {max_pos}")
        print(f"   - 当前位置: {positions[0]}")
        
        # 手动计算百分比
        if max_pos == min_pos:
            manual_percent = 0
        else:
            manual_percent = (positions[0] - min_pos) / (max_pos - min_pos) * 100
            manual_percent = max(0, min(100, manual_percent))
        
        print(f"   - 手动计算的百分比: {manual_percent:.2f}%")
        
        # 使用夹爪的方法
        gripper_percent = gripper.get_percent()
        print(f"   - get_percent() 返回值: {gripper_percent}")
        
        if gripper_percent is None:
            print(f"   ⚠️  get_percent() 返回 None（读取失败）")
        elif abs(gripper_percent - manual_percent) > 0.01:
            print(f"   ⚠️  百分比不匹配！")
            print(f"      - 手动计算: {manual_percent:.2f}%")
            print(f"      - get_percent(): {gripper_percent:.2f}%")
        else:
            print(f"   ✅ 百分比计算正确")
            
    except Exception as e:
        print(f"   ❌ 百分比转换异常: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. 测试角度转换
    print(f"\n6. 测试角度转换")
    try:
        angle = gripper.get_angle()
        print(f"   - get_angle() 返回值: {angle}")
        if angle is not None:
            print(f"   - 角度: {angle:.2f}°")
    except Exception as e:
        print(f"   ❌ 角度转换异常: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. 测试移动（可选）
    print(f"\n7. 测试移动（可选）")
    try:
        current_percent = gripper.get_percent()
        if current_percent is not None:
            print(f"   - 当前位置: {current_percent:.2f}%")
            
            # 如果当前位置接近0，尝试移动到50%
            if current_percent < 10:
                print(f"   - 尝试移动到 50%...")
                if gripper.set_percent(50, wait=True):
                    time.sleep(0.5)
                    new_percent = gripper.get_percent()
                    print(f"   - 移动后位置: {new_percent:.2f}%")
                else:
                    print(f"   - ❌ 移动失败")
            
            # 如果当前位置接近100，尝试移动到0%
            elif current_percent > 90:
                print(f"   - 尝试移动到 0%...")
                if gripper.set_percent(0, wait=True):
                    time.sleep(0.5)
                    new_percent = gripper.get_percent()
                    print(f"   - 移动后位置: {new_percent:.2f}%")
                else:
                    print(f"   - ❌ 移动失败")
            else:
                print(f"   - 跳过移动测试（当前位置在中间）")
    except Exception as e:
        print(f"   ❌ 移动测试异常: {e}")
        import traceback
        traceback.print_exc()
    
    # 8. 断开连接
    print(f"\n8. 断开连接")
    try:
        gripper.disconnect()
        print(f"   ✅ 已断开连接")
    except Exception as e:
        print(f"   ⚠️  断开连接异常: {e}")
    
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='夹爪调试测试')
    parser.add_argument('--side', '-s', choices=['left', 'right'], default='left',
                        help='选择左夹爪或右夹爪')
    args = parser.parse_args()
    
    test_gripper_connection(args.side)

