#!/usr/bin/env python3
"""
扫描 Feetech 舵机 ID
"""

import sys
sys.path.append(".")
from scservo_sdk import *

def scan_servos(port='/dev/ttyACM0', baudrate=1000000, id_range=range(1, 20)):
    """扫描舵机ID"""
    
    portHandler = PortHandler(port)
    packetHandler = sms_sts(portHandler)
    
    if not portHandler.openPort():
        print(f"❌ 无法打开串口: {port}")
        return []
    
    if not portHandler.setBaudRate(baudrate):
        print(f"❌ 无法设置波特率: {baudrate}")
        return []
    
    print(f"扫描串口: {port}, 波特率: {baudrate}")
    print(f"扫描ID范围: {id_range.start} - {id_range.stop - 1}")
    print("-" * 40)
    
    found = []
    for servo_id in id_range:
        # 尝试读取位置
        pos, result, error = packetHandler.read2ByteTxRx(
            servo_id, SMS_STS_PRESENT_POSITION_L
        )
        if result == COMM_SUCCESS:
            print(f"✅ 找到舵机 ID={servo_id}, 位置={pos}")
            found.append((servo_id, pos))
        else:
            sys.stdout.write(f"\r   扫描 ID={servo_id}...")
            sys.stdout.flush()
    
    print("\r" + " " * 30)  # 清除扫描行
    portHandler.closePort()
    
    if found:
        print("-" * 40)
        print(f"共找到 {len(found)} 个舵机")
    else:
        print("❌ 未找到任何舵机")
        print("\n请检查:")
        print("  - 舵机是否通电")
        print("  - 连接线是否正确")
        print("  - 尝试其他波特率 (如 500000, 115200)")
    
    return found


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='扫描 Feetech 舵机')
    parser.add_argument('--port', '-p', default='/dev/ttyACM0')
    parser.add_argument('--baudrate', '-b', type=int, default=1000000)
    parser.add_argument('--max-id', '-m', type=int, default=20)
    
    args = parser.parse_args()
    
    scan_servos(args.port, args.baudrate, range(1, args.max_id + 1))


