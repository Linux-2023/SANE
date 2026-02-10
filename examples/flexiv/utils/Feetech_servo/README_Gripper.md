# Gripper 夹爪控制类使用说明

## 概述

`Gripper` 类用于控制 Feetech 舵机驱动的夹爪，支持百分比控制和角度控制。

**特点：**
- 自动从配置文件加载标定数据
- 支持左/右夹爪独立控制
- 支持百分比控制 (0-100%)
- 支持角度控制 (0° ~ 最大角度)
- 自动限制输入范围，防止超限
- 支持 `with` 语句自动管理连接

## 快速开始

### 1. 标定夹爪（首次使用需要）

```bash
# 标定左夹爪
python3 gripper_calibration.py --id 7 --side left

# 标定右夹爪  
python3 gripper_calibration.py --id 1 --side right
```

标定时：
1. 程序会关闭舵机力矩
2. 手动将夹爪移到**关闭位置**，按 Enter
3. 手动将夹爪移到**打开位置**，按 Enter
4. 数据自动保存到 `gripper_config.json`

### 2. 基本使用

```python
from gripper import Gripper

# 使用 with 语句（推荐，自动管理连接）
with Gripper(side='left') as gripper:
    gripper.open()   # 打开夹爪
    gripper.close()  # 关闭夹爪
```

## API 参考

### 初始化

```python
Gripper(side='left', config_file=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `side` | str | `'left'` | 夹爪位置：`'left'` 或 `'right'` |
| `config_file` | str | `None` | 配置文件路径，默认使用 `gripper_config.json` |

### 连接管理

```python
gripper.connect()      # 连接夹爪，返回 bool
gripper.disconnect()   # 断开连接
gripper.is_connected   # 属性，是否已连接
```

### 运动控制

#### 打开/关闭

```python
gripper.open(speed=1500, acc=50, wait=True)   # 完全打开
gripper.close(speed=1500, acc=50, wait=True)  # 完全关闭
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `speed` | int | 1500 | 速度 (0-4095) |
| `acc` | int | 50 | 加速度 (0-255) |
| `wait` | bool | True | 是否等待移动完成 |

#### 百分比控制

```python
gripper.set_percent(percent, speed=1500, acc=50, wait=False)
```

| 参数 | 说明 |
|------|------|
| `percent` | 目标开度 0-100（0=关闭，100=打开）|

**示例：**
```python
gripper.set_percent(0)    # 完全关闭
gripper.set_percent(50)   # 开到一半
gripper.set_percent(100)  # 完全打开
gripper.set_percent(150)  # 自动限制为 100
```

#### 角度控制

```python
gripper.set_angle(angle, speed=1500, acc=50, wait=False)
```

| 参数 | 说明 |
|------|------|
| `angle` | 目标夹爪角度（0 ~ max_angle）|

**注意：** 夹爪角度 = 电机角度 × 2

**示例：**
```python
gripper.set_angle(0)    # 完全关闭
gripper.set_angle(30)   # 开到 30°
gripper.set_angle(62)   # 完全打开（约62°）
gripper.set_angle(200)  # 自动限制为最大角度
```

### 状态读取

```python
gripper.get_percent()   # 返回当前开度百分比 (0-100)，失败返回 None
gripper.get_angle()     # 返回当前夹爪角度，失败返回 None
gripper.max_angle       # 属性，最大夹爪角度（约62°）
gripper.servo_id        # 属性，舵机ID
gripper.is_moving()     # 是否在移动中
gripper.get_status()    # 返回完整状态字典
```

**get_status() 返回值：**
```python
{
    'side': 'left',
    'servo_id': 7,
    'connected': True,
    'percent': 50.0,
    'angle': 31.0,
    'max_angle': 62.1,
    'moving': False
}
```

### 等待完成

```python
gripper.wait_until_done(timeout=5.0)  # 等待移动完成，返回是否成功
```

## 使用示例

### 示例 1：基本开合控制

```python
from gripper import Gripper

with Gripper(side='left') as gripper:
    print(f"最大角度: {gripper.max_angle:.1f}°")
    
    # 打开
    gripper.open()
    print(f"当前角度: {gripper.get_angle():.1f}°")
    
    # 关闭
    gripper.close()
    print(f"当前角度: {gripper.get_angle():.1f}°")
```

### 示例 2：精确角度控制

```python
from gripper import Gripper

with Gripper(side='right') as gripper:
    # 移动到指定角度
    for angle in [0, 15, 30, 45, 60]:
        gripper.set_angle(angle, wait=True)
        print(f"目标: {angle}°, 实际: {gripper.get_angle():.1f}°")
```

### 示例 3：百分比控制

```python
from gripper import Gripper

with Gripper(side='left') as gripper:
    # 渐进式打开
    for percent in range(0, 101, 10):
        gripper.set_percent(percent, speed=2000, wait=True)
        print(f"开度: {percent}%")
```

### 示例 4：同时控制两个夹爪

```python
from gripper import Gripper

left = Gripper(side='left')
right = Gripper(side='right')

left.connect()
right.connect()

try:
    # 同时打开
    left.open(wait=False)
    right.open(wait=False)
    
    # 等待都完成
    left.wait_until_done()
    right.wait_until_done()
    
    # 同时关闭
    left.close(wait=False)
    right.close(wait=True)
finally:
    left.disconnect()
    right.disconnect()
```

### 示例 5：手动管理连接

```python
from gripper import Gripper

gripper = Gripper(side='left')

if gripper.connect():
    print("连接成功")
    gripper.open()
    gripper.close()
    gripper.disconnect()
else:
    print("连接失败")
```

## 命令行使用

```bash
# 显示夹爪信息
python3 gripper.py --side left

# 打开夹爪
python3 gripper.py --side left --action open

# 关闭夹爪
python3 gripper.py --side right --action close

# 移动到指定角度
python3 gripper.py --side left --action 30
```

## 配置文件

标定数据保存在 `gripper_config.json`：

```json
{
  "port": "/dev/ttyACM0",
  "baudrate": 1000000,
  "grippers": {
    "left": {
      "servo_id": 7,
      "min_position": 1404,
      "max_position": 1051,
      "calibration_time": "2026-01-05 09:44:46"
    },
    "right": {
      "servo_id": 1,
      "min_position": 3671,
      "max_position": 3321,
      "calibration_time": "2026-01-05 09:45:25"
    }
  }
}
```

## 角度说明

| 概念 | 说明 |
|------|------|
| 电机角度 | 舵机实际转动的角度 |
| 夹爪角度 | 夹爪开合的角度 = 电机角度 × 2 |
| min_position | 电机关闭位置（夹爪角度 0°）|
| max_position | 电机打开位置（夹爪角度 max_angle）|

## 安全特性

- **自动范围限制**：角度和百分比超出范围时自动限制
  - `set_percent(150)` → 实际执行 100%
  - `set_angle(-10)` → 实际执行 0°
- **速度限制**：speed 自动限制在 0-4095
- **加速度限制**：acc 自动限制在 0-255

## 错误处理

```python
from gripper import Gripper

try:
    gripper = Gripper(side='left')
except FileNotFoundError as e:
    print("配置文件不存在，请先标定")
except ValueError as e:
    print("配置文件中没有该夹爪的数据")
```

## 文件结构

```
Feetech_servo/
├── gripper.py              # 夹爪控制类
├── gripper_calibration.py  # 标定程序
├── gripper_config.json     # 标定数据（自动生成）
├── scan_servo.py           # 舵机扫描工具
└── scservo_sdk/            # Feetech SDK
```


