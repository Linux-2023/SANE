#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import minimalmodbus
import serial.tools.list_ports

class GripperController:
    def __init__(self, port=None, baudrate=115200, bytesize=8, parity='N', stopbits=1, timeout=0.5, slave_id=1):
        if port is None:
            port = self._detect_port()
        self.instrument = minimalmodbus.Instrument(port, slave_id)
        self.instrument.serial.baudrate = baudrate
        self.instrument.serial.bytesize = bytesize
        self.instrument.serial.parity = parity
        self.instrument.serial.stopbits = stopbits
        self.instrument.serial.timeout = timeout
        self.instrument.close_port_after_each_call = True

        self.slave_id = slave_id

        self.ENABLE_REG = 0x0100
        self.TEMP_POSITION_H = 0x0102
        self.TEMP_POSITION_L = 0x0103
        self.TEMP_SPEED = 0x0104
        self.TEMP_FORCE = 0x0105
        self.TEMP_ACCEL = 0x0106
        self.TEMP_DECEL = 0x0107
        self.TEMP_TRIGGER = 0x0108

        self.MAX_SPEED = 0x0305
        self.MAX_FORCE = 0x0306
        self.MAX_ACCEL = 0x0307
        self.MAX_DECEL = 0x0308

        self.MULTI_MODE = 0x0110
        self.MULTI_START = 0x0111
        self.MULTI_END = 0x0112
        self.MULTI_TRIGGER = 0x0117
        self.MULTI_PAUSE = 0x0118

        self.STATUS_FORCE_REACHED = 0x0601
        self.STATUS_POS_REACHED = 0x0602
        self.STATUS_SPEED_REACHED = 0x0603
        self.STATUS_READY = 0x0604
        self.STATUS_POS_H = 0x0609
        self.STATUS_POS_L = 0x060A
        self.STATUS_SPEED = 0x060B
        self.STATUS_CURRENT = 0x060C
        self.STATUS_ALARM = 0x0612


        self.default_speed = 100  # range: 0-100
        self.default_force = 100  # range: 0-100
        self.default_accel = 100  # range: 0-100
        self.default_decel = 100  # range: 0-100
        self.default_max_speed = 10000 # range: 1-10000
        self.default_max_force = 2000 # range: 1-2000
        self.default_max_accel = 10000 # range: 1-10000
        self.default_max_decel = 10000 # range: 1-10000

        self.set_max_parameters()
        self.set_motion_parameters()

    def _detect_port(self):
        ports = [p.device for p in serial.tools.list_ports.comports() if 'USB Serial' in p.description or 'RS485' in p.description]
        if not ports:
            raise RuntimeError('未检测到RS485端口')
        return ports[0]

    def connect(self):
        try:
            self.read_status()
            return True
        except:
            return False

    def disconnect(self):
        self.instrument.serial.close()

    def enable_gripper(self, enable=True):
        value = 1 if enable else 0
        self.instrument.write_register(self.ENABLE_REG, value)

    def set_position_32bit(self, position):
        high_word = (position >> 16) & 0xFFFF
        low_word = position & 0xFFFF
        self.instrument.write_register(self.TEMP_POSITION_H, high_word)
        self.instrument.write_register(self.TEMP_POSITION_L, low_word)

    def get_current_position(self):
        high_word = self.instrument.read_register(self.STATUS_POS_H)
        low_word = self.instrument.read_register(self.STATUS_POS_L)
        return (high_word << 16) | low_word

    # 设置最大速度、最大力、最大加速度、最大减速度
    def set_max_parameters(self, speed=None, force=None, accel=None, decel=None):
        params = {
            'speed': speed or self.default_max_speed,
            'force': force or self.default_max_force,   
            'accel': accel or self.default_max_accel,
            'decel': decel or self.default_max_decel
        }
        self.instrument.write_register(self.MAX_SPEED, params['speed'])
        self.instrument.write_register(self.MAX_FORCE, params['force'])
        self.instrument.write_register(self.MAX_ACCEL, params['accel'])
        self.instrument.write_register(self.MAX_DECEL, params['decel'])

    def set_motion_parameters(self, speed=None, force=None, accel=None, decel=None):
        params = {
            'speed': speed or self.default_speed,
            'force': force or self.default_force,   
            'accel': accel or self.default_accel,
            'decel': decel or self.default_decel
        }
        self.instrument.write_register(self.TEMP_SPEED, params['speed'])
        self.instrument.write_register(self.TEMP_FORCE, params['force'])
        self.instrument.write_register(self.TEMP_ACCEL, params['accel'])
        self.instrument.write_register(self.TEMP_DECEL, params['decel'])

    def trigger_motion(self):
        self.instrument.write_register(self.TEMP_TRIGGER, 1)

    def move_to_position(self, position, speed=None, force=None, accel=None, decel=None):
        step2 = time.time()
        self.set_position_32bit(position)
        print(f"set_position_32bit耗时: {time.time() - step2:.6f} s")
        step3 = time.time()
        self.trigger_motion()
        print(f"trigger_motion耗时: {time.time() - step3:.6f} s")

    def open_gripper(self, position=0, speed=1000, force=100):
        self.move_to_position(position, speed, force)

    def close_gripper(self, position=6000, speed=500, force=60):
        self.move_to_position(position, speed, force)

    def read_status(self):
        status = {}
        status_regs = [
            (self.STATUS_FORCE_REACHED, '力矩到达'),
            (self.STATUS_POS_REACHED, '位置到达'),
            (self.STATUS_READY, '准备完成'),
            (self.STATUS_ALARM, '报警信息')
        ]
        for reg_addr, name in status_regs:
            status[name] = self.instrument.read_register(reg_addr)
        return status

    def wait_for_motion_complete(self, timeout=10):
        start_time = time.time()
        while time.time() - start_time < timeout:
            ready = self.instrument.read_register(self.STATUS_READY)
            if ready == 1:
                print(f"Motion complete:{time.time() - start_time} s")
                return True
            time.sleep(0.1)
        return False

    def trigger_multi_position(self, start_pos=1, end_pos=2, mode=0):
        self.instrument.write_register(self.MULTI_MODE, mode)
        self.instrument.write_register(self.MULTI_START, start_pos)
        self.instrument.write_register(self.MULTI_END, end_pos)
        self.instrument.write_register(self.MULTI_TRIGGER, 1)

    def read_alarms(self):
        alarm_code = self.instrument.read_register(self.STATUS_ALARM)
        alarms = []
        if alarm_code & 0x01: alarms.append('过温警报')
        if alarm_code & 0x02: alarms.append('堵转警报')
        if alarm_code & 0x04: alarms.append('超速警报')
        if alarm_code & 0x08: alarms.append('初始化故障警报')
        if alarm_code & 0x10: alarms.append('超限检测警报')
        if alarm_code & 0x20: alarms.append('夹取掉落警报')
        return {'code': alarm_code, 'alarms': alarms}

def main():
    gripper = GripperController(port=None, baudrate=115200, slave_id=1, timeout=1)
    if not gripper.connect():
        return
    gripper.enable_gripper(True)
    time.sleep(0.5)
    start_position = gripper.get_current_position()
    print(f"Start position: {start_position}")
    gripper.move_to_position(position=6000, speed=100, force=100,accel=100,decel=100)
    start_flag = False
    start_time = time.time()
    while not start_flag:
        current_position = gripper.get_current_position()
        if abs(current_position - start_position) > 1:
            start_flag = True
            print(f"Start motion, current position: {current_position}")
            print(f"Start time: {time.time() - start_time} s")
        time.sleep(0.01)
    time.sleep(1)
    start_position = gripper.get_current_position()
    print(f"Start position: {start_position}")
    step_start_time = time.time()
    gripper.move_to_position(position=0, speed=100, force=100, accel=100, decel=100)
    print(f"Step duration: {time.time() - step_start_time} s")
    start_flag = False
    start_time = time.time()
    while not start_flag:
        current_position = gripper.get_current_position()
        if abs(current_position - start_position) > 1:
            start_flag = True
            print(f"Start motion, current position: {current_position}")
            print(f"Start time: {time.time() - start_time} s")
        time.sleep(0.01)
    gripper.wait_for_motion_complete()
    current_position = gripper.get_current_position()
    print(f"Current position: {current_position}")
    gripper.enable_gripper(False)
    gripper.disconnect()

if __name__ == '__main__':
    main() 