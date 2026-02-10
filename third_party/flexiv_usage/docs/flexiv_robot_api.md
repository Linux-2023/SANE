# FlexivRobot 接口文档

> 版本: 1.0.0  
> 更新日期: 2024-12

`FlexivRobot` 是对 `flexivrdk` 库的高层封装，提供更简洁易用的机器人控制接口。

---

## 目录

- [安装依赖](#安装依赖)
- [快速开始](#快速开始)
- [类 FlexivRobot](#类-flexivrobot)
  - [构造函数](#构造函数)
  - [初始化方法](#初始化方法)
  - [状态查询方法](#状态查询方法)
  - [阻塞式运动方法](#阻塞式运动方法)
  - [非阻塞式运动方法](#非阻塞式运动方法)
  - [直接发送命令](#直接发送命令)
  - [夹爪控制](#夹爪控制)
  - [工具方法](#工具方法)
- [数据类](#数据类)
  - [MotionParams](#motionparams)
  - [ForceControlFrame](#forcecontrolframe)
- [工具函数](#工具函数)
- [使用示例](#使用示例)
- [注意事项](#注意事项)

---

## 安装依赖

```bash
pip install flexivrdk numpy
```

---

## 快速开始

```python
from flexiv_robot import FlexivRobot
import math

# 创建并初始化机器人
robot = FlexivRobot("Rizon4s-063239")

# 获取当前状态
print(f"关节角: {robot.get_joint_positions()}")
print(f"TCP 位姿: {robot.get_tcp_pose()}")

# 关节运动
joints = robot.get_joint_positions()
joints[5] += math.radians(30)  # J6 转 30 度
robot.move_joint(joints)

# 笛卡尔运动
pose = robot.get_tcp_pose()
pose[2] += 0.05  # Z 方向上移 5cm
robot.move_cartesian(pose)

# 停止
robot.stop()
```

---

## 类 FlexivRobot

### 构造函数

```python
FlexivRobot(
    robot_sn: str,
    auto_init: bool = True,
    motion_params: Optional[MotionParams] = None,
    home_joints: Optional[Union[List[float], np.ndarray]] = None,
    verbose: bool = True
)
```

创建 Flexiv 机器人控制实例。

**参数:**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `robot_sn` | `str` | - | 机器人序列号，如 `"Rizon4s-063239"` |
| `auto_init` | `bool` | `True` | 是否自动执行初始化流程 |
| `motion_params` | `MotionParams` | `None` | 运动参数配置，`None` 使用默认值 |
| `home_joints` | `list/ndarray` | `None` | Home 位置的关节角度 [rad]，`None` 则使用初始化时的当前位置 |
| `verbose` | `bool` | `True` | 是否打印详细日志 |

**示例:**

```python
import math

# 自动初始化（推荐），Home 位置为初始化时的当前位置
robot = FlexivRobot("Rizon4s-063239")

# 指定 Home 位置（7 个关节的角度，单位：rad）
home = [0, -math.pi/4, 0, math.pi/2, 0, math.pi/4, 0]
robot = FlexivRobot("Rizon4s-063239", home_joints=home)

# 手动初始化
robot = FlexivRobot("Rizon4s-063239", auto_init=False)
robot.initialize()

# 自定义运动参数
params = MotionParams(joint_max_vel=1.5, control_frequency=100)
robot = FlexivRobot("Rizon4s-063239", motion_params=params)
```

---

### 初始化方法

#### `initialize(timeout_sec: float = 60.0) -> bool`

执行完整的初始化流程：连接 → 清除故障 → 使能 → 等待就绪。

**参数:**
- `timeout_sec`: 等待就绪的超时时间 [s]

**返回:** 是否初始化成功

**示例:**

```python
robot = FlexivRobot("Rizon4s-063239", auto_init=False)
if robot.initialize(timeout_sec=30):
    print("初始化成功")
else:
    print("初始化失败")
```

---

#### `is_ready() -> bool`

检查机器人是否已就绪可操作。

**返回:** `True` 已就绪，`False` 未就绪

---

#### `stop()`

停止机器人运动并切换到 IDLE 模式。

---

### 状态查询方法

#### `get_joint_positions() -> np.ndarray`

获取当前关节角度。

**返回:** 关节角度数组，单位 `[rad]`，shape=`(n_joints,)`

**示例:**

```python
joints = robot.get_joint_positions()
print(f"J1: {joints[0]} rad, {math.degrees(joints[0])}°")
```

---

#### `get_tcp_pose() -> np.ndarray`

获取当前 TCP 位姿。

**返回:** 位姿数组 `[x, y, z, qw, qx, qy, qz]`，单位 `[m]` 和 `[]`

**示例:**

```python
pose = robot.get_tcp_pose()
print(f"位置: {pose[:3]}")
print(f"姿态: {pose[3:7]}")
```

---

#### `get_tcp_position() -> np.ndarray`

获取当前 TCP 位置。

**返回:** 位置数组 `[x, y, z]`，单位 `[m]`

---

#### `get_tcp_quaternion() -> np.ndarray`

获取当前 TCP 姿态（四元数）。

**返回:** 四元数数组 `[qw, qx, qy, qz]`

---

#### `get_flange_pose() -> np.ndarray`

获取当前法兰位姿。

**返回:** 位姿数组 `[x, y, z, qw, qx, qy, qz]`

---

#### `get_joint_torques() -> np.ndarray`

获取当前关节力矩。

**返回:** 关节力矩数组，单位 `[Nm]`

---

#### `get_external_wrench(frame: str = "world") -> np.ndarray`

获取外部力/力矩。

**参数:**
- `frame`: 参考坐标系，`"world"` 或 `"tcp"`

**返回:** 力/力矩数组 `[fx, fy, fz, mx, my, mz]`，单位 `[N, Nm]`

**示例:**

```python
wrench_world = robot.get_external_wrench("world")
wrench_tcp = robot.get_external_wrench("tcp")
```

---

#### `get_robot_info() -> flexivrdk.RobotInfo`

获取机器人信息对象。

---

#### `get_dof() -> int`

获取机器人自由度。

**返回:** 自由度数量（通常为 7）

---

### 阻塞式运动方法

阻塞式方法会等待运动完成后才返回。

#### `move_joint(target_joints, timeout, tolerance) -> bool`

阻塞式关节运动。

```python
move_joint(
    target_joints: Union[List[float], np.ndarray],
    timeout: float = 30.0,
    tolerance: Optional[float] = None
) -> bool
```

**参数:**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `target_joints` | `list/ndarray` | - | 目标关节角度 `[rad]` |
| `timeout` | `float` | 30.0 | 超时时间 `[s]` |
| `tolerance` | `float` | None | 到达判定阈值 `[rad]`，None 使用默认值 |

**返回:** 是否成功到达目标

**示例:**

```python
import math

current = robot.get_joint_positions()
target = current.copy()
target[5] += math.radians(30)  # J6 转 30 度

success = robot.move_joint(target, timeout=20.0)
if success:
    print("运动完成")
```

---

#### `move_cartesian(target_pose, timeout, position_tolerance, orientation_tolerance) -> bool`

阻塞式笛卡尔位置运动。

```python
move_cartesian(
    target_pose: Union[List[float], np.ndarray],
    timeout: float = 30.0,
    position_tolerance: Optional[float] = None,
    orientation_tolerance: Optional[float] = None
) -> bool
```

**参数:**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `target_pose` | `list/ndarray` | - | 目标位姿 `[x,y,z,qw,qx,qy,qz]` |
| `timeout` | `float` | 30.0 | 超时时间 `[s]` |
| `position_tolerance` | `float` | None | 位置误差阈值 `[m]` |
| `orientation_tolerance` | `float` | None | 姿态误差阈值 |

**返回:** 是否成功到达目标

**示例:**

```python
current = robot.get_tcp_pose()
target = current.copy()
target[2] += 0.05  # Z 方向上移 5cm

success = robot.move_cartesian(target)
```

---

#### `move_cartesian_force(target_pose, target_wrench, force_axes, force_frame, timeout, position_tolerance) -> bool`

阻塞式力位混合运动。

```python
move_cartesian_force(
    target_pose: Union[List[float], np.ndarray],
    target_wrench: Union[List[float], np.ndarray],
    force_axes: List[bool],
    force_frame: ForceControlFrame = ForceControlFrame.WORLD,
    timeout: float = 30.0,
    position_tolerance: Optional[float] = None
) -> bool
```

**参数:**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `target_pose` | `list/ndarray` | - | 目标位姿 `[x,y,z,qw,qx,qy,qz]` |
| `target_wrench` | `list/ndarray` | - | 目标力/力矩 `[fx,fy,fz,mx,my,mz]` |
| `force_axes` | `List[bool]` | - | 力控轴使能 `[X,Y,Z,Rx,Ry,Rz]` |
| `force_frame` | `ForceControlFrame` | WORLD | 力控参考坐标系 |
| `timeout` | `float` | 30.0 | 超时时间 `[s]` |
| `position_tolerance` | `float` | None | 位置误差阈值 `[m]` |

**返回:** 是否成功完成

**示例:**

```python
from flexiv_robot import ForceControlFrame

pose = robot.get_tcp_pose()

# Z 轴力控（10N 下压），X/Y 轴位置保持
success = robot.move_cartesian_force(
    target_pose=pose,
    target_wrench=[0, 0, 10, 0, 0, 0],
    force_axes=[False, False, True, False, False, False],
    force_frame=ForceControlFrame.WORLD,
    timeout=10.0
)
```

---

### 非阻塞式运动方法

非阻塞式方法会立即返回，运动在后台线程执行。

#### `move_joint_async(target_joints, callback, timeout, tolerance) -> Thread`

非阻塞式关节运动。

```python
move_joint_async(
    target_joints: Union[List[float], np.ndarray],
    callback: Optional[callable] = None,
    timeout: float = 30.0,
    tolerance: Optional[float] = None
) -> threading.Thread
```

**参数:**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `target_joints` | `list/ndarray` | - | 目标关节角度 `[rad]` |
| `callback` | `callable` | None | 完成回调函数，签名 `(success: bool) -> None` |
| `timeout` | `float` | 30.0 | 超时时间 `[s]` |
| `tolerance` | `float` | None | 到达判定阈值 `[rad]` |

**返回:** 运动线程对象

**示例:**

```python
def on_complete(success):
    print(f"运动完成: {'成功' if success else '失败'}")

thread = robot.move_joint_async(target, callback=on_complete)

# 可以同时做其他事情
while robot.is_moving():
    print(f"当前位置: {robot.get_joint_positions()}")
    time.sleep(0.1)
```

---

#### `move_cartesian_async(target_pose, callback, timeout, position_tolerance, orientation_tolerance) -> Thread`

非阻塞式笛卡尔运动。

```python
move_cartesian_async(
    target_pose: Union[List[float], np.ndarray],
    callback: Optional[callable] = None,
    timeout: float = 30.0,
    position_tolerance: Optional[float] = None,
    orientation_tolerance: Optional[float] = None
) -> threading.Thread
```

---

#### `move_cartesian_force_async(...) -> Thread`

非阻塞式力位混合运动。

```python
move_cartesian_force_async(
    target_pose: Union[List[float], np.ndarray],
    target_wrench: Union[List[float], np.ndarray],
    force_axes: List[bool],
    force_frame: ForceControlFrame = ForceControlFrame.WORLD,
    callback: Optional[callable] = None,
    timeout: float = 30.0,
    position_tolerance: Optional[float] = None
) -> threading.Thread
```

---

#### `wait_motion_complete(timeout: Optional[float] = None) -> bool`

等待当前异步运动完成。

**参数:**
- `timeout`: 等待超时时间 `[s]`，`None` 无限等待

**返回:** 运动是否已结束

---

#### `cancel_motion()`

取消当前异步运动。

---

#### `is_moving() -> bool`

检查是否有运动正在执行。

---

### 直接发送命令

这些方法发送单次命令，不等待到达。用户需要自己实现循环调用。

#### `send_joint_position(target_joints, target_vel, target_acc)`

直接发送一次关节位置命令。

```python
send_joint_position(
    target_joints: Union[List[float], np.ndarray],
    target_vel: Optional[Union[List[float], np.ndarray]] = None,
    target_acc: Optional[Union[List[float], np.ndarray]] = None
)
```

**参数:**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `target_joints` | `list/ndarray` | - | 目标关节角度 `[rad]` |
| `target_vel` | `list/ndarray` | None | 目标关节速度 `[rad/s]`，None 则为 0 |
| `target_acc` | `list/ndarray` | None | 目标关节加速度 `[rad/s²]`，None 则为 0 |

**示例:**

```python
# 自定义控制循环
for i in range(500):
    time.sleep(0.02)  # 50Hz
    robot.send_joint_position(target)
```

---

#### `send_cartesian_pose(target_pose, target_wrench)`

直接发送一次笛卡尔位姿命令。

```python
send_cartesian_pose(
    target_pose: Union[List[float], np.ndarray],
    target_wrench: Optional[Union[List[float], np.ndarray]] = None
)
```

---

### 夹爪控制

夹爪控制方法支持阻塞和非阻塞两种模式。通过 `wait` 参数控制：
- `wait=True`（默认）：等待夹爪运动完成后返回
- `wait=False`：发送命令后立即返回，可通过 `is_gripper_moving()` 检查状态

#### `gripper_enable(gripper_name: str, switch_tool: bool = True)`

使能夹爪。

**参数:**
- `gripper_name`: 夹爪名称，如 `"Flexiv-GN01"`
- `switch_tool`: 是否同时切换机器人工具到夹爪（用于重力补偿和 TCP 位置）

**示例:**

```python
robot.gripper_enable("Flexiv-GN01")
```

---

#### `gripper_disable()`

禁用当前夹爪。

---

#### `gripper_init(wait: bool = True, timeout: float = 10.0) -> bool`

手动初始化夹爪（某些夹爪需要）。

**参数:**
- `wait`: 是否等待初始化完成
- `timeout`: 等待超时时间 [s]

**返回:** 是否初始化成功

---

#### `gripper_move(width, velocity, force_limit, wait, timeout) -> bool`

移动夹爪到指定开口宽度。

```python
gripper_move(
    width: float,
    velocity: Optional[float] = None,
    force_limit: Optional[float] = None,
    wait: bool = True,
    timeout: float = 10.0
) -> bool
```

**参数:**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `width` | `float` | - | 目标开口宽度 [m] |
| `velocity` | `float` | None | 移动速度 [m/s]，None 则使用最大速度的 50% |
| `force_limit` | `float` | None | 最大接触力 [N]，None 则使用最大力的 50% |
| `wait` | `bool` | True | 是否等待运动完成 |
| `timeout` | `float` | 10.0 | 等待超时时间 [s] |

**返回:** 是否成功到达目标（wait=False 时始终返回 True）

**示例:**

```python
# 阻塞式：移动到 5cm 开口
robot.gripper_move(0.05)

# 非阻塞式
robot.gripper_move(0.05, wait=False)
while robot.is_gripper_moving():
    print(f"当前宽度: {robot.get_gripper_width()*1000:.1f}mm")
    time.sleep(0.1)
```

---

#### `gripper_grasp(force, wait, timeout) -> bool`

力控夹取（需要夹爪支持力控）。

**参数:**
- `force`: 目标夹持力 [N]，正值为闭合力，负值为张开力
- `wait`: 是否等待夹取稳定
- `timeout`: 等待超时时间 [s]

**返回:** 是否成功

**示例:**

```python
# 以 10N 的力夹取
robot.gripper_grasp(10.0)
```

---

#### `gripper_open(velocity, wait, timeout) -> bool`

完全张开夹爪。

**参数:**
- `velocity`: 移动速度 [m/s]，None 则使用最大速度的 50%
- `wait`: 是否等待运动完成
- `timeout`: 等待超时时间 [s]

**示例:**

```python
robot.gripper_open()
```

---

#### `gripper_close(force_limit, velocity, wait, timeout) -> bool`

完全闭合夹爪。

**参数:**
- `force_limit`: 最大接触力 [N]，None 则使用最大力的 50%
- `velocity`: 移动速度 [m/s]，None 则使用最大速度的 50%
- `wait`: 是否等待运动完成
- `timeout`: 等待超时时间 [s]

**示例:**

```python
# 以最大 20N 的力闭合
robot.gripper_close(force_limit=20.0)
```

---

#### `gripper_stop()`

停止夹爪运动并保持当前位置。

---

#### `is_gripper_moving() -> bool`

检查夹爪是否正在运动。

**返回:** True 正在运动，False 已停止

---

#### `wait_gripper(timeout: float = 10.0) -> bool`

等待夹爪运动完成。

**参数:**
- `timeout`: 超时时间 [s]

**返回:** 是否在超时前完成

---

#### `get_gripper_width() -> float`

获取当前夹爪开口宽度。

**返回:** 开口宽度 [m]

---

#### `get_gripper_force() -> float`

获取当前夹爪夹持力。

**返回:** 夹持力 [N]，正值为张开力，负值为闭合力

---

#### `get_gripper_states() -> GripperStates`

获取夹爪完整状态。

**返回:** `GripperStates` 对象，包含：
- `width`: 开口宽度 [m]
- `force`: 夹持力 [N]
- `is_moving`: 是否在运动

---

#### `get_gripper_params() -> GripperParams`

获取夹爪参数。

**返回:** `GripperParams` 对象，包含：
- `name`: 夹爪名称
- `min_width`, `max_width`: 宽度范围 [m]
- `min_vel`, `max_vel`: 速度范围 [m/s]
- `min_force`, `max_force`: 力范围 [N]

---

#### `is_gripper_enabled() -> bool`

检查夹爪是否已使能。

---

### 工具方法

#### `set_motion_params(params: MotionParams)`

设置运动参数。

---

#### `switch_mode(mode: flexivrdk.Mode)`

切换控制模式。

---

#### `execute_primitive(name: str, params: dict)`

执行原语。

**示例:**

```python
robot.execute_primitive("Home", {})

robot.execute_primitive("MoveL", {
    "target": flexivrdk.Coord([0.5, 0, 0.3], [180, 0, 180], ["WORLD", "WORLD_ORIGIN"]),
    "vel": 0.3
})
```

---

#### `execute_plan(name: str, wait: bool = True)`

执行计划。（需要配合示教器使用，暂不用管）

**示例:**

```python
robot.execute_plan("PLAN-Home", wait=True)
```

---

#### `go_home(wait: bool = True, timeout: float = 30.0) -> bool`

回到 Home 位置（初始化时设定的关节角度）。

**参数:**
- `wait`: 是否等待运动完成
- `timeout`: 超时时间 [s]（仅在 wait=True 时有效）

**返回:** 是否成功到达 Home 位置（wait=False 时始终返回 True）

**示例:**

```python
# 回到 Home 并等待
success = robot.go_home(wait=True)

# 非阻塞回 Home
robot.go_home(wait=False)
```

---

#### `set_home_joints(home_joints: Union[List[float], np.ndarray])`

设置 Home 位置的关节角度。

**参数:**
- `home_joints`: Home 位置的关节角度 [rad]

**示例:**

```python
import math

# 设置自定义 Home 位置
home = [0, -math.pi/4, 0, math.pi/2, 0, math.pi/4, 0]
robot.set_home_joints(home)
```

---

#### `set_current_as_home()`

将当前关节角度设置为 Home 位置。

**示例:**

```python
# 手动将机器人移到期望位置后
robot.set_current_as_home()
```

---

#### `get_home_joints() -> Optional[np.ndarray]`

获取 Home 位置的关节角度。

**返回:** Home 关节角度 [rad]，未设置时返回 None

**示例:**

```python
home = robot.get_home_joints()
print(f"Home 位置: {np.degrees(home)}°")
```

---

#### `zero_ft_sensor(timeout: float = 30.0)`

清零力/力矩传感器。

---

#### `robot -> flexivrdk.Robot`

属性，获取底层 Robot 对象用于高级操作。

---

## 数据类

### MotionParams

运动参数配置。

```python
@dataclass
class MotionParams:
    # 关节运动参数
    joint_max_vel: float = 1.0       # 最大关节速度 [rad/s]
    joint_max_acc: float = 1.5       # 最大关节加速度 [rad/s²]
    
    # 笛卡尔运动参数
    cart_max_linear_vel: float = 0.5   # 最大线速度 [m/s]
    cart_max_angular_vel: float = 1.0  # 最大角速度 [rad/s]
    cart_max_linear_acc: float = 2.0   # 最大线加速度 [m/s²]
    cart_max_angular_acc: float = 5.0  # 最大角加速度 [rad/s²]
    
    # 控制参数
    control_frequency: int = 50      # 控制频率 [Hz]
    
    # 到达判定阈值
    joint_tolerance: float = 0.01    # 关节误差阈值 [rad]
    position_tolerance: float = 0.001  # 位置误差阈值 [m]
    orientation_tolerance: float = 0.01  # 姿态误差阈值
```

**使用示例:**

```python
params = MotionParams(
    joint_max_vel=1.5,
    cart_max_linear_vel=0.3,
    control_frequency=100
)
robot = FlexivRobot("Rizon4s-063239", motion_params=params)
```

---

### ForceControlFrame

力控制参考坐标系枚举。

```python
class ForceControlFrame(Enum):
    WORLD = "world"  # 世界坐标系（固定）
    TCP = "tcp"      # TCP 坐标系（随末端移动）
```

---

## 工具函数

### `load_robot_config(path: str) -> dict`

从 YAML 文件加载完整的机器人配置。

**文件格式:**
```yaml
name: Rizon4s-063239 # arm serial number
gripper_name: Flexiv-GN01
```

**返回:** 配置字典，包含 `robot_sn` 和可选的 `gripper_name`

**示例:**

```python
from flexiv_robot import load_robot_config

config = load_robot_config("robot.yaml")
robot_sn = config["robot_sn"]
gripper_name = config.get("gripper_name")  # 可能不存在

# 使用配置初始化
robot = FlexivRobot(robot_sn)
if gripper_name:
    robot.gripper_enable(gripper_name)
```

---

### `load_robot_sn_from_yaml(path: str) -> str`

从 YAML 文件加载机器人序列号（向后兼容）。

**文件格式:**
```yaml
name: Rizon4s-063239 # serial number
```

**示例:**

```python
robot_sn = load_robot_sn_from_yaml("robot.yaml")
```

---

### `quat_to_rotation_matrix(qw, qx, qy, qz) -> np.ndarray`

四元数转 3x3 旋转矩阵。

---

### `pose_to_matrix(pose) -> np.ndarray`

位姿转 4x4 变换矩阵。

**示例:**

```python
pose = robot.get_tcp_pose()
T = pose_to_matrix(pose)  # 4x4 齐次变换矩阵
```

---

## 使用示例

### 示例 1: 基本运动控制

```python
from flexiv_robot import FlexivRobot
import math

# 初始化
robot = FlexivRobot("Rizon4s-063239")

# 关节运动
joints = robot.get_joint_positions()
joints[5] += math.radians(30)
robot.move_joint(joints)

# 笛卡尔运动
pose = robot.get_tcp_pose()
pose[2] += 0.05
robot.move_cartesian(pose)

robot.stop()
```

### 示例 2: 非阻塞运动 + 状态监控

```python
from flexiv_robot import FlexivRobot
import time
import math

robot = FlexivRobot("Rizon4s-063239")

target = robot.get_joint_positions()
target[5] += math.radians(30)

# 启动非阻塞运动
robot.move_joint_async(target)

# 监控运动过程
while robot.is_moving():
    current = robot.get_joint_positions()
    error = abs(current[5] - target[5])
    print(f"误差: {math.degrees(error):.2f}°")
    time.sleep(0.1)

print("运动完成")
robot.stop()
```

### 示例 3: 力位混合控制

```python
from flexiv_robot import FlexivRobot, ForceControlFrame

robot = FlexivRobot("Rizon4s-063239")

# 清零力传感器
robot.zero_ft_sensor()

pose = robot.get_tcp_pose()

# Z 轴 10N 下压，X/Y 保持位置
robot.move_cartesian_force(
    target_pose=pose,
    target_wrench=[0, 0, 10, 0, 0, 0],
    force_axes=[False, False, True, False, False, False],
    force_frame=ForceControlFrame.WORLD,
    timeout=10.0
)

robot.stop()
```

### 示例 4: 自定义控制循环

```python
from flexiv_robot import FlexivRobot
import time
import math
import numpy as np

robot = FlexivRobot("Rizon4s-063239")

# 获取初始位姿
init_pose = robot.get_tcp_pose()

# 正弦轨迹参数
amplitude = 0.03  # 3cm
frequency = 0.5   # 0.5Hz
duration = 10.0   # 10秒

start_time = time.time()
while time.time() - start_time < duration:
    t = time.time() - start_time
    
    # 计算目标位姿
    target = init_pose.copy()
    target[1] += amplitude * math.sin(2 * math.pi * frequency * t)
    
    # 发送命令
    robot.send_cartesian_pose(target)
    
    time.sleep(0.02)  # 50Hz

robot.stop()
```

### 示例 5: 夹爪控制

```python
from flexiv_robot import FlexivRobot
import time

robot = FlexivRobot("Rizon4s-063239")

# 使能夹爪
robot.gripper_enable("Flexiv-GN01")

# 查看夹爪参数
params = robot.get_gripper_params()
print(f"夹爪宽度范围: {params.min_width*1000:.1f} - {params.max_width*1000:.1f} mm")
print(f"夹爪力范围: {params.min_force:.1f} - {params.max_force:.1f} N")

# 阻塞式：完全张开
robot.gripper_open()
print(f"当前宽度: {robot.get_gripper_width()*1000:.1f} mm")

# 阻塞式：移动到指定宽度
robot.gripper_move(0.05)  # 5cm
print(f"当前宽度: {robot.get_gripper_width()*1000:.1f} mm")

# 非阻塞式：闭合夹爪
robot.gripper_close(wait=False)
while robot.is_gripper_moving():
    states = robot.get_gripper_states()
    print(f"宽度: {states.width*1000:.1f}mm, 力: {states.force:.1f}N")
    time.sleep(0.1)

# 力控夹取
robot.gripper_grasp(force=15.0)  # 15N 夹持力
print(f"当前夹持力: {robot.get_gripper_force():.1f} N")

# 张开并禁用
robot.gripper_open()
robot.gripper_disable()

robot.stop()
```

### 示例 6: 机械臂与夹爪协同控制

```python
from flexiv_robot import FlexivRobot
import time

robot = FlexivRobot("Rizon4s-063239")
robot.gripper_enable("Flexiv-GN01")

# 1. 移动到抓取位置上方
pick_pose = robot.get_tcp_pose()
pick_pose[2] += 0.1  # 上方 10cm
robot.move_cartesian(pick_pose)

# 2. 张开夹爪
robot.gripper_open()

# 3. 下降到抓取位置
pick_pose[2] -= 0.1
robot.move_cartesian(pick_pose)

# 4. 闭合夹爪抓取
robot.gripper_close(force_limit=20.0)

# 5. 提起
pick_pose[2] += 0.1
robot.move_cartesian(pick_pose)

# 6. 移动到放置位置（示例：X方向移动 20cm）
place_pose = pick_pose.copy()
place_pose[0] += 0.2
robot.move_cartesian(place_pose)

# 7. 下降并释放
place_pose[2] -= 0.1
robot.move_cartesian(place_pose)
robot.gripper_open()

# 8. 提起并回到初始位置
place_pose[2] += 0.1
robot.move_cartesian(place_pose)

robot.gripper_disable()
robot.stop()
```

---

## 注意事项

### 安全相关

1. **急停按钮**: 始终将急停按钮放在手边
2. **工作空间**: 确保机器人工作范围内无人和障碍物
3. **速度限制**: 测试时使用较低的速度参数
4. **力控制**: 力控制测试时从小力值开始

### 使用建议

1. **初始化检查**: 运动前检查 `is_ready()` 返回 `True`
2. **异常处理**: 使用 try-except 捕获运动异常
3. **超时设置**: 根据运动距离合理设置超时时间
4. **频率选择**: 控制频率建议 50-100Hz

### 单位约定

| 物理量 | 单位 |
|--------|------|
| 位置 | 米 (m) |
| 关节角度 | 弧度 (rad) |
| 速度 | m/s 或 rad/s |
| 力 | 牛顿 (N) |
| 力矩 | 牛·米 (Nm) |
| 时间 | 秒 (s) |

### 坐标系

- **世界坐标系 (WORLD)**: 固定在机器人基座
- **TCP 坐标系**: 固定在末端执行器，随机器人运动

---

## 测试

运行单元测试：

```bash
# 交互式测试
python test/test_interactive.py
```

---

*文档版本: 1.0.0*

