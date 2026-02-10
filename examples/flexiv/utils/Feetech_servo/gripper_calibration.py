#!/usr/bin/env python3
"""
夹爪标定脚本
用于手动标定夹爪的最小值和最大值位置
"""

import sys
import time
import json
import os

sys.path.append(".")
from scservo_sdk import *
from scservo_sdk.sms_sts import SMS_STS_TORQUE_ENABLE, SMS_STS_PRESENT_POSITION_L


CONFIG_FILE = 'gripper_config.json'


class GripperCalibrator:
    """夹爪标定器"""
    
    def __init__(self, port='/dev/ttyACM0', baudrate=1000000, servo_id=1, side='left'):
        self.servo_id = servo_id
        self.port = port
        self.baudrate = baudrate
        self.side = side  # 'left' 或 'right'
        
        self.portHandler = PortHandler(port)
        self.packetHandler = sms_sts(self.portHandler)
        
        self.min_position = None
        self.max_position = None
        
    def connect(self) -> bool:
        """连接舵机"""
        if not self.portHandler.openPort():
            print(f"❌ 无法打开串口: {self.port}")
            return False
        
        if not self.portHandler.setBaudRate(self.baudrate):
            print(f"❌ 无法设置波特率: {self.baudrate}")
            return False
        
        print(f"✅ 已连接到 {self.port}")
        return True
    
    def disconnect(self):
        """断开连接"""
        self.portHandler.closePort()
        print("已断开连接")
    
    def disable_torque(self):
        """关闭力矩，允许手动移动"""
        result, error = self.packetHandler.write1ByteTxRx(
            self.servo_id, SMS_STS_TORQUE_ENABLE, 0
        )
        if result == COMM_SUCCESS:
            print("✅ 力矩已关闭，现在可以手动移动夹爪")
            return True
        else:
            print(f"⚠️  关闭力矩失败 (result={result}, error={error})")
            print("   尝试继续标定（某些舵机可直接手动移动）...")
            return True  # 仍然继续，因为有些舵机可以直接手动移动
    
    def enable_torque(self):
        """开启力矩"""
        result, error = self.packetHandler.write1ByteTxRx(
            self.servo_id, SMS_STS_TORQUE_ENABLE, 1
        )
        if result == COMM_SUCCESS:
            print("✅ 力矩已开启")
        return True  # 不管成功与否都返回 True
    
    def read_position(self) -> int:
        """读取当前位置"""
        pos, result, error = self.packetHandler.read2ByteTxRx(
            self.servo_id, SMS_STS_PRESENT_POSITION_L
        )
        if result == COMM_SUCCESS:
            return pos
        return -1
    
    def test_communication(self) -> bool:
        """测试与舵机的通信"""
        print(f"\n正在测试与舵机 ID={self.servo_id} 的通信...")
        pos = self.read_position()
        if pos >= 0:
            print(f"✅ 通信正常，当前位置: {pos}")
            return True
        else:
            print(f"❌ 无法读取舵机位置，请检查:")
            print(f"   - 舵机ID是否正确 (当前: {self.servo_id})")
            print(f"   - 串口是否正确 (当前: {self.port})")
            print(f"   - 波特率是否正确 (当前: {self.baudrate})")
            print(f"   - 舵机是否通电")
            return False
    
    def run_calibration(self):
        """运行标定流程"""
        side_name = "左" if self.side == "left" else "右"
        print("\n" + "=" * 50)
        print(f"       夹爪标定程序 - {side_name}夹爪")
        print("=" * 50)
        
        if not self.connect():
            return False
        
        # 先测试通信
        if not self.test_communication():
            self.disconnect()
            return False
        
        if not self.disable_torque():
            self.disconnect()
            return False
        
        print("\n📋 标定说明:")
        print("   1. 手动移动夹爪到 关闭（最小）位置")
        print("   2. 按 Enter 记录最小值")
        print("   3. 手动移动夹爪到 打开（最大）位置")
        print("   4. 按 Enter 记录最大值")
        print("   5. 按 q 退出")
        print("\n当前位置实时显示中...\n")
        
        try:
            # 记录最小值
            print("-" * 40)
            print("第一步: 将夹爪移动到 关闭（最小）位置")
            print("-" * 40)
            self.min_position = self._record_position("最小值")
            if self.min_position is None:
                print("标定已取消")
                return False
            
            # 记录最大值
            print("\n" + "-" * 40)
            print("第二步: 将夹爪移动到 打开（最大）位置")
            print("-" * 40)
            self.max_position = self._record_position("最大值")
            if self.max_position is None:
                print("标定已取消")
                return False
            
            # 显示结果
            self._show_results()
            
            # 保存结果
            self._save_calibration()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n标定已取消")
            return False
        finally:
            self.enable_torque()
            self.disconnect()
    
    def _record_position(self, label: str) -> int:
        """交互式记录位置"""
        import select
        import termios
        import tty
        
        # 保存终端设置
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # 设置终端为非阻塞模式
            tty.setcbreak(sys.stdin.fileno())
            
            print(f"移动夹爪后按 Enter 记录{label}，按 q 退出")
            print()
            
            while True:
                # 读取当前位置
                pos = self.read_position()
                if pos >= 0:
                    # 使用回车符覆盖显示
                    sys.stdout.write(f"\r   当前位置: {pos:4d}    ")
                    sys.stdout.flush()
                
                # 检查键盘输入（非阻塞）
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if char == '\n' or char == '\r':
                        print(f"\n\n✅ {label}已记录: {pos}")
                        return pos
                    elif char.lower() == 'q':
                        print("\n")
                        return None
                        
        finally:
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def _show_results(self):
        """显示标定结果"""
        print("\n" + "=" * 50)
        print("       标定结果")
        print("=" * 50)
        print(f"  最小位置 (关闭): {self.min_position}")
        print(f"  最大位置 (打开): {self.max_position}")
        print(f"  位置范围: {abs(self.max_position - self.min_position)}")
        print("=" * 50)
    
    def _save_calibration(self):
        """保存标定数据到文件"""
        # 读取现有配置（如果存在）
        config = {'port': self.port, 'baudrate': self.baudrate, 'grippers': {}}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
            except:
                pass
        
        # 更新对应夹爪的配置
        config['port'] = self.port
        config['baudrate'] = self.baudrate
        config['grippers'][self.side] = {
            'servo_id': self.servo_id,
            'min_position': self.min_position,
            'max_position': self.max_position,
            'calibration_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存到配置文件
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        side_name = "左" if self.side == "left" else "右"
        print(f"\n💾 {side_name}夹爪标定数据已保存到: {CONFIG_FILE}")
        
        # 显示如何使用
        print("\n📖 使用方法:")
        print("-" * 40)
        print(f"  python3 gripper_test.py --side {self.side} --action open")
        print(f"  python3 gripper_test.py --side {self.side} --action close")
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='夹爪标定工具')
    parser.add_argument('--port', '-p', default='/dev/ttyACM0', 
                        help='串口设备路径 (默认: /dev/ttyACM0)')
    parser.add_argument('--baudrate', '-b', type=int, default=1000000,
                        help='波特率 (默认: 1000000)')
    parser.add_argument('--id', '-i', type=int, default=1,
                        help='舵机ID (默认: 1)')
    parser.add_argument('--side', '-s', choices=['left', 'right'], default='left',
                        help='夹爪位置: left(左) 或 right(右) (默认: left)')
    
    args = parser.parse_args()
    
    calibrator = GripperCalibrator(
        port=args.port,
        baudrate=args.baudrate,
        servo_id=args.id,
        side=args.side
    )
    
    success = calibrator.run_calibration()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

