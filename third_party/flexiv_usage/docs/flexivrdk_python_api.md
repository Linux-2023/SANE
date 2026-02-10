# Flexiv RDK v1.7.0 Python API 文档

本文档基于 Flexiv RDK v1.7.0 版本，描述了用于控制 Flexiv 机器人的 Python API。

---

## 目录

1. [概述](#概述)
2. [常量定义](#常量定义)
3. [枚举类型](#枚举类型)
4. [数据结构](#数据结构)
5. [核心类](#核心类)
   - [Robot 类](#robot-类)
   - [Gripper 类](#gripper-类)
   - [Model 类](#model-类)
   - [Tool 类](#tool-类)

---

## 概述

Flexiv RDK (Robot Development Kit) 是一套用于控制 Flexiv 机器人的开发套件。Python API 提供了与 C++ API 相同的功能，包括：

- 机器人状态监控
- 多种控制模式（关节控制、笛卡尔控制等）
- 夹爪控制
- 计划和原语执行
- 机器人动力学模型

**导入模块：**

```python
import flexivrdk
```

---

## 常量定义

| 常量名 | 值 | 说明 |
|--------|-----|------|
| `kCartDoF` | 6 | 笛卡尔空间自由度 |
| `kSerialJointDoF` | 7 | 串联机器人关节自由度 |
| `kPoseSize` | 7 | 位姿尺寸（位置3 + 四元数4） |
| `kIOPorts` | 18 | IO端口数量 |
| `kMaxExtAxes` | 6 | 最大外部轴数量 |

---

## 枚举类型

### Mode - 机器人控制模式

机器人需要切换到正确的控制模式后才能执行相应的命令。

| 模式 | 说明 |
|------|------|
| `Mode.UNKNOWN` | 未知模式 |
| `Mode.IDLE` | 空闲模式，机器人可操作时的默认模式 |
| `Mode.RT_JOINT_TORQUE` | 实时关节力矩控制模式 |
| `Mode.RT_JOINT_IMPEDANCE` | 实时关节阻抗控制模式 |
| `Mode.NRT_JOINT_IMPEDANCE` | 非实时关节阻抗控制模式 |
| `Mode.RT_JOINT_POSITION` | 实时关节位置控制模式 |
| `Mode.NRT_JOINT_POSITION` | 非实时关节位置控制模式 |
| `Mode.NRT_PLAN_EXECUTION` | 非实时计划执行模式 |
| `Mode.NRT_PRIMITIVE_EXECUTION` | 非实时原语执行模式 |
| `Mode.RT_CARTESIAN_MOTION_FORCE` | 实时笛卡尔运动力控制模式 |
| `Mode.NRT_CARTESIAN_MOTION_FORCE` | 非实时笛卡尔运动力控制模式 |
| `Mode.NRT_SUPER_PRIMITIVE` | 非实时超级原语模式（试验性） |

### OperationalStatus - 机器人操作状态

| 状态 | 说明 |
|------|------|
| `OperationalStatus.UNKNOWN` | 未知状态 |
| `OperationalStatus.READY` | 就绪，可以操作 |
| `OperationalStatus.BOOTING` | 系统启动中，请等待 |
| `OperationalStatus.ESTOP_NOT_RELEASED` | 急停未释放 |
| `OperationalStatus.NOT_ENABLED` | 未使能，需调用 `Enable()` |
| `OperationalStatus.RELEASING_BRAKE` | 正在释放制动器，请等待 |
| `OperationalStatus.MINOR_FAULT` | 发生轻微故障，可调用 `ClearFault()` 尝试清除 |
| `OperationalStatus.CRITICAL_FAULT` | 发生严重故障，可调用 `ClearFault()` 尝试清除 |
| `OperationalStatus.IN_REDUCED_STATE` | 处于降速状态 |
| `OperationalStatus.IN_RECOVERY_STATE` | 处于恢复状态 |
| `OperationalStatus.IN_MANUAL_MODE` | 处于手动模式，需切换到自动(远程)模式 |
| `OperationalStatus.IN_AUTO_MODE` | 处于常规自动模式，需切换到自动(远程)模式 |

### CoordType - 坐标系类型

| 类型 | 说明 |
|------|------|
| `CoordType.WORLD` | 世界坐标系（固定） |
| `CoordType.TCP` | TCP 坐标系（随末端执行器移动） |

---

## 数据结构

### RobotInfo - 机器人信息

机器人的一般信息结构体。

| 属性 | 类型 | 说明 |
|------|------|------|
| `serial_num` | str | 机器人序列号 |
| `software_ver` | str | 软件版本 |
| `model_name` | str | 机器人型号名称 |
| `license_type` | str | 许可类型 |
| `DoF_e` | int | 外部轴自由度 |
| `DoF_m` | int | 主机器人自由度 |
| `DoF` | int | 总自由度 |
| `K_x_nom` | List[float] | 笛卡尔空间标称刚度 [Nm/rad] |
| `K_q_nom` | List[float] | 关节空间标称刚度 [Nm/rad] |
| `q_min` | List[float] | 关节位置下限 [rad] |
| `q_max` | List[float] | 关节位置上限 [rad] |
| `dq_max` | List[float] | 关节速度上限 [rad/s] |
| `tau_max` | List[float] | 关节力矩上限 [Nm] |

### RobotStates - 机器人状态

机器人在关节空间和笛卡尔空间的状态数据。

| 属性 | 类型 | 说明 |
|------|------|------|
| `q` | List[float] | 测量的关节位置 [rad] |
| `theta` | List[float] | 测量的电机位置 [rad] |
| `dq` | List[float] | 测量的关节速度 [rad/s] |
| `dtheta` | List[float] | 测量的电机速度 [rad/s] |
| `tau` | List[float] | 测量的关节力矩 [Nm] |
| `tau_des` | List[float] | 期望的关节力矩 [Nm] |
| `tau_dot` | List[float] | 关节力矩变化率 [Nm/s] |
| `tau_ext` | List[float] | 估计的外部力矩 [Nm] |
| `tcp_pose` | List[float] | TCP位姿 [x,y,z,qw,qx,qy,qz] [m]:[无单位] |
| `tcp_vel` | List[float] | TCP速度 [vx,vy,vz,wx,wy,wz] [m/s]:[rad/s] |
| `flange_pose` | List[float] | 法兰位姿 [x,y,z,qw,qx,qy,qz] [m]:[无单位] |
| `ft_sensor_raw` | List[float] | 原始力/力矩传感器读数 [Fx,Fy,Fz,Mx,My,Mz] [N]:[Nm] |
| `ext_wrench_in_tcp` | List[float] | TCP坐标系中的外部力/力矩 [N]:[Nm] |
| `ext_wrench_in_world` | List[float] | 世界坐标系中的外部力/力矩 [N]:[Nm] |
| `ext_wrench_in_tcp_raw` | List[float] | TCP坐标系中的原始外部力/力矩 [N]:[Nm] |
| `ext_wrench_in_world_raw` | List[float] | 世界坐标系中的原始外部力/力矩 [N]:[Nm] |

### RobotEvent - 机器人事件

机器人事件信息结构体。

| 属性 | 类型 | 说明 |
|------|------|------|
| `level` | int | 事件级别 (UNKNOWN=0, INFO, WARNING, ERROR, CRITICAL) |
| `id` | int | 事件ID |
| `description` | str | 事件描述 |
| `consequences` | str | 事件后果 |
| `probable_causes` | str | 可能的原因 |
| `recommended_actions` | str | 建议的操作 |
| `timestamp` | datetime | 事件时间戳 |

### PlanInfo - 计划信息

正在执行的原语/计划的信息。

| 属性 | 类型 | 说明 |
|------|------|------|
| `pt_name` | str | 原语名称 |
| `node_name` | str | 节点名称 |
| `node_path` | str | 节点路径 |
| `node_path_time_period` | str | 节点路径时间周期 |
| `node_path_number` | str | 节点路径编号 |
| `assigned_plan_name` | str | 分配的计划名称 |
| `velocity_scale` | float | 速度缩放因子 |
| `waiting_for_step` | bool | 是否等待步进 |

### GripperParams - 夹爪参数

夹爪参数数据结构。

| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | str | 夹爪名称 |
| `min_width` | float | 最小手指开合宽度 [m] |
| `max_width` | float | 最大手指开合宽度 [m] |
| `min_vel` | float | 最小手指移动速度 [m/s] |
| `max_vel` | float | 最大手指移动速度 [m/s] |
| `min_force` | float | 最小抓取力 [N] |
| `max_force` | float | 最大抓取力 [N] |

### GripperStates - 夹爪状态

夹爪状态数据结构。

| 属性 | 类型 | 说明 |
|------|------|------|
| `width` | float | 测量的手指开合宽度 [m] |
| `force` | float | 测量的手指力。正：开力，负：闭力 [N] |
| `is_moving` | bool | 夹爪手指是否正在移动 |

### ToolParams - 工具参数

机器人工具参数数据结构。

| 属性 | 类型 | 说明 |
|------|------|------|
| `mass` | float | 总质量 [kg] |
| `CoM` | List[float] | 机器人法兰坐标系中的质心 [x,y,z] [m] |
| `inertia` | List[float] | 质心处的惯量 [Ixx,Iyy,Izz,Ixy,Ixz,Iyz] [kg·m²] |
| `tcp_location` | List[float] | TCP在法兰坐标系中的位置和方向 [x,y,z,qw,qx,qy,qz] [m]:[无单位] |

### SafetyLimits - 安全限制

可配置的机器人安全限制数据结构。

| 属性 | 类型 | 说明 |
|------|------|------|
| `q_min` | List[float] | 关节位置下限 [rad] |
| `q_max` | List[float] | 关节位置上限 [rad] |
| `dq_max_normal` | List[float] | 正常状态下的关节速度上限 [rad/s] |
| `dq_max_reduced` | List[float] | 降速状态下的关节速度上限 [rad/s] |

### JPos - 关节位置

Flexiv Elements 中自定义数据类型 "JPOS" 的数据结构。

| 属性 | 类型 | 说明 |
|------|------|------|
| `q_m` | List[float] | 主机器人关节位置 [rad] |
| `q_e` | List[float] | 外部轴关节位置 [rad] |

### Coord - 坐标

Flexiv Elements 中自定义数据类型 "COORD" 的数据结构。

| 属性 | 类型 | 说明 |
|------|------|------|
| `position` | List[float] | 位置 [x,y,z] [m] |
| `orientation` | List[float] | 方向（欧拉角） [rx,ry,rz] [rad] |
| `ref_frame` | List[str] | 参考坐标系 |
| `ref_q_m` | List[float] | 参考主机器人关节位置 |
| `ref_q_e` | List[float] | 参考外部轴关节位置 |

---

## 核心类

### Robot 类

控制机器人的主接口，包含多个功能类别和后台服务。

#### 构造函数

```python
Robot(robot_sn: str, network_interface_whitelist: List[str] = [], verbose: bool = True)
```

**参数：**
- `robot_sn`: 要连接的机器人序列号。接受格式："Rizon 4s-123456" 或 "Rizon4s-123456"
- `network_interface_whitelist`: 限制用于尝试建立连接的网络接口。通过关联的IPv4地址定义。空列表表示尝试所有可用接口
- `verbose`: 启用/禁用信息和警告打印

**异常：**
- `std::invalid_argument`: robot_sn 格式无效
- `std::runtime_error`: 初始化序列失败
- `std::logic_error`: 机器人缺少有效的RDK许可证、与RDK库版本不兼容或不支持的机器人型号

#### 状态查询方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `connected()` | bool | 是否已与机器人建立连接 |
| `info()` | RobotInfo | 获取机器人一般信息 |
| `mode()` | Mode | 获取当前控制模式 |
| `states()` | RobotStates | 获取当前机器人状态数据 |
| `stopped()` | bool | 机器人是否已完全停止 |
| `operational()` | bool | 机器人是否可操作（使能、制动器释放、自动模式、无故障、非降速状态） |
| `operational_status()` | OperationalStatus | 获取当前操作状态 |
| `busy()` | bool | 机器人是否忙碌（执行计划、原语、运动等） |
| `fault()` | bool | 机器人是否处于故障状态 |
| `reduced()` | bool | 机器人是否处于降速状态 |
| `recovery()` | bool | 机器人是否处于恢复状态 |
| `estop_released()` | bool | 急停是否已释放 |
| `enabling_button_pressed()` | bool | 使能按钮是否被按下 |
| `event_log()` | List[RobotEvent] | 获取自上次成功实例化以来的机器人事件日志 |
| `digital_inputs()` | List[bool] | 获取所有数字输入端口的当前读数（18个端口） |

#### 控制方法

##### 使能与模式切换

```python
def Enable() -> None
```
[阻塞] 使能机器人。如果急停已释放且无故障，机器人将释放制动器并在几秒后变为可操作状态。

```python
def Brake(engage: bool) -> None
```
[阻塞] 在正常操作期间强制接合或释放机器人制动器。仅限医疗型号机器人在静止时使用。

```python
def SwitchMode(mode: Mode) -> None
```
[阻塞] 将机器人切换到新的控制模式并等待模式切换完成。

```python
def Stop() -> None
```
[阻塞] 停止机器人并将控制模式切换到 IDLE。

```python
def ClearFault(timeout_sec: int = 30) -> bool
```
[阻塞] 尝试清除机器人的轻微或严重故障，无需重新上电。返回是否成功清除故障。

```python
def RunAutoRecovery() -> None
```
[阻塞] 运行自动恢复，将超出允许位置范围的关节带回允许范围。

##### 全局变量

```python
def SetGlobalVariables(global_vars: Dict[str, FlexivDataTypes]) -> None
```
[阻塞] 设置机器人中已存在的全局变量的值。

```python
def global_variables() -> Dict[str, FlexivDataTypes]
```
[阻塞] 获取所有全局变量及其当前值。

##### 外部轴

```python
def LockExternalAxes(toggle: bool) -> None
```
[阻塞] 在原语执行、直接关节控制和直接笛卡尔控制模式下锁定/解锁外部轴。

##### 计划执行

```python
def ExecutePlan(index: int, continue_exec: bool = False, block_until_started: bool = True) -> None
def ExecutePlan(name: str, continue_exec: bool = False, block_until_started: bool = True) -> None
```
[阻塞] 通过索引或名称执行计划。

**参数：**
- `index/name`: 计划索引或名称
- `continue_exec`: RDK程序关闭或连接断开时是否继续执行计划
- `block_until_started`: 是否等待计划开始执行后再返回

**适用模式：** NRT_PLAN_EXECUTION

```python
def PausePlan(toggle: bool) -> None
```
[阻塞] 暂停或恢复当前计划的执行。

```python
def plan_list() -> List[str]
```
[阻塞] 获取所有可用计划的列表。

```python
def plan_info() -> PlanInfo
```
[阻塞] 获取正在执行的计划的详细信息。

```python
def SetBreakpointMode(is_enabled: bool) -> None
```
[阻塞] 启用或禁用计划执行期间的断点模式。

```python
def StepBreakpoint() -> None
```
[阻塞] 如果启用了断点模式，步进到下一个断点。

```python
def SetVelocityScale(velocity_scale: int) -> None
```
[阻塞] 设置计划和原语执行期间机器人运动的整体速度缩放。取值范围：[1, 100]。

##### 原语执行

```python
def ExecutePrimitive(primitive_name: str, input_params: Dict[str, FlexivDataTypes], block_until_started: bool = True) -> None
```
[阻塞] 通过名称和参数执行原语。

**参数：**
- `primitive_name`: 原语名称
- `input_params`: 输入参数字典
- `block_until_started`: 是否等待原语开始执行后再返回

**适用模式：** NRT_PRIMITIVE_EXECUTION

```python
def primitive_states() -> Dict[str, FlexivDataTypes]
```
[阻塞] 获取正在执行的原语的状态参数名称和值。

##### 关节控制

```python
def StreamJointTorque(torques: List[float], enable_gravity_comp: bool = True, enable_soft_limits: bool = True) -> None
```
[非阻塞] 连续流式发送关节力矩命令。

**参数：**
- `torques`: 目标关节力矩 [Nm]
- `enable_gravity_comp`: 启用机器人内置重力补偿
- `enable_soft_limits`: 启用关节位置和速度的软限制

**适用模式：** RT_JOINT_TORQUE

```python
def StreamJointPosition(positions: List[float], velocities: List[float], accelerations: List[float]) -> None
```
[非阻塞] 连续流式发送关节位置、速度和加速度命令。

**适用模式：** RT_JOINT_IMPEDANCE, RT_JOINT_POSITION

```python
def SendJointPosition(positions: List[float], velocities: List[float], accelerations: List[float], max_vel: List[float], max_acc: List[float]) -> None
```
[非阻塞] 离散发送关节位置命令。机器人内部运动生成器会平滑离散命令。

**适用模式：** NRT_JOINT_IMPEDANCE, NRT_JOINT_POSITION

```python
def SetJointImpedance(K_q: List[float], Z_q: List[float] = []) -> None
```
[阻塞] 设置关节阻抗控制模式下的阻抗属性。

**参数：**
- `K_q`: 关节刚度 [Nm/rad]
- `Z_q`: 关节阻尼比（默认使用标称值）

##### 笛卡尔控制

```python
def StreamCartesianMotionForce(pose: List[float], wrench: List[float] = [], velocity: List[float] = [], acceleration: List[float] = []) -> None
```
[非阻塞] 连续流式发送笛卡尔运动和/或力命令。

**参数：**
- `pose`: 目标TCP位姿 [x,y,z,qw,qx,qy,qz] [m]:[无单位]
- `wrench`: 目标力/力矩 [Fx,Fy,Fz,Mx,My,Mz] [N]:[Nm]
- `velocity`: 目标TCP速度 [vx,vy,vz,wx,wy,wz] [m/s]:[rad/s]
- `acceleration`: 目标TCP加速度

**适用模式：** RT_CARTESIAN_MOTION_FORCE

```python
def SendCartesianMotionForce(pose: List[float], wrench: List[float] = [], max_linear_vel: float = 0.5, max_angular_vel: float = 1.0, max_linear_acc: float = 2.0, max_angular_acc: float = 5.0) -> None
```
[非阻塞] 离散发送笛卡尔运动和/或力命令。

**适用模式：** NRT_CARTESIAN_MOTION_FORCE

```python
def SetCartesianImpedance(K_x: List[float], Z_x: List[float] = [0.7]*6) -> None
```
[阻塞] 设置笛卡尔阻抗控制的阻抗属性。

**参数：**
- `K_x`: 笛卡尔刚度 [N/m, N/m, N/m, Nm/rad, Nm/rad, Nm/rad]
- `Z_x`: 笛卡尔阻尼比

```python
def SetMaxContactWrench(max_wrench: List[float]) -> None
```
[阻塞] 设置笛卡尔运动力控制模式下运动控制部分的最大接触力/力矩。

```python
def SetNullSpacePosture(ref_positions: List[float]) -> None
```
[阻塞] 设置笛卡尔运动力控制模式下零空间姿态控制模块的参考关节位置。

```python
def SetNullSpaceObjectives(linear_manipulability: float = 0.0, angular_manipulability: float = 0.0, ref_positions_tracking: float = 0.5) -> None
```
[阻塞] 设置计算机器人零空间姿态时三个优化目标的权重。

```python
def SetForceControlAxis(enabled_axes: List[bool], max_linear_vel: List[float] = [1.0]*3) -> None
```
[阻塞] 设置笛卡尔运动力控制模式下启用力控制的轴。

**参数：**
- `enabled_axes`: 6个布尔值，True表示该轴启用力控制
- `max_linear_vel`: 力控制轴的最大线速度 [m/s]

```python
def SetForceControlFrame(root_coord: CoordType, T_in_root: List[float] = [0,0,0,1,0,0,0]) -> None
```
[阻塞] 设置力控制的参考坐标系。

```python
def SetPassiveForceControl(is_enabled: bool) -> None
```
[阻塞] 启用或禁用被动力控制。启用时使用开环力控制，禁用时使用闭环力控制。

##### 数字IO

```python
def SetDigitalOutputs(digital_outputs: Dict[int, bool]) -> None
```
[阻塞] 设置一个或多个数字输出端口。

**参数：**
- `digital_outputs`: 端口索引到值的映射。索引0-15对应控制箱端口，16-17对应腕部连接器内部端口

---

### Gripper 类

控制安装在机器人上的夹爪的接口。

#### 构造函数

```python
Gripper(robot: Robot)
```

**参数：**
- `robot`: flexiv.rdk.Robot 实例的引用

#### 方法

```python
def Enable(name: str) -> None
```
[阻塞] 启用指定的夹爪作为机器人设备。

**注意：** 同一时间只能有一个启用的夹爪，启用新夹爪前需先调用 `Disable()`。

```python
def Disable() -> None
```
[阻塞] 禁用当前启用的夹爪。

```python
def Init() -> None
```
[阻塞] 手动触发已启用夹爪的初始化。对于上电后自动初始化的夹爪，此步骤不需要。

**警告：** 此函数不会等待初始化序列完成。

```python
def Grasp(force: float) -> None
```
[阻塞] 使用直接力控制进行抓取。

**参数：**
- `force`: 目标抓取力。正值：闭合力，负值：张开力。范围：[min_force, max_force] [N]

```python
def Move(width: float, velocity: float, force_limit: float) -> None
```
[阻塞] 使用位置控制移动夹爪手指。

**参数：**
- `width`: 目标开合宽度 [m]
- `velocity`: 闭合/张开速度，不能为0 [m/s]
- `force_limit`: 运动过程中的最大接触力 [N]

```python
def Stop() -> None
```
[阻塞] 停止夹爪并保持当前手指宽度。

```python
def params() -> GripperParams
```
[非阻塞] 获取当前启用夹爪的参数。

```python
def states() -> GripperStates
```
[非阻塞] 获取当前启用夹爪的状态数据。

---

### Model 类

获取机器人模型数据的接口，包括运动学和动力学。

**注意：** 需要 RDK 专业许可证。

#### 构造函数

```python
Model(robot: Robot, gravity_vector: List[float] = [0.0, 0.0, -9.81])
```

**参数：**
- `robot`: flexiv.rdk.Robot 实例的引用
- `gravity_vector`: 世界坐标系中的重力向量 [m/s²]

#### 方法

```python
def Reload() -> None
```
[阻塞] 使用从连接机器人同步的最新数据重新加载本地存储的机器人模型参数。工具模型也会同步。

**警告：** 每当连接的机器人发生物理变化（如添加或更换工具）时，必须使用此函数手动刷新本地存储的机器人模型参数。

```python
def Update(positions: List[float], velocities: List[float]) -> None
```
[非阻塞] 更新本地存储的机器人模型的配置（姿态）。

**参数：**
- `positions`: 当前关节位置 [rad]
- `velocities`: 当前关节速度 [rad/s]

```python
def J(link_name: str) -> numpy.ndarray
```
[非阻塞] 计算指定坐标系相对于世界坐标系的雅可比矩阵。

**参数：**
- `link_name`: 链接名称。可用链接：{"base_link", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "flange"}，如果安装了工具还包括 "tool"

**返回：** 雅可比矩阵 (6×n)

```python
def dJ(link_name: str) -> numpy.ndarray
```
[非阻塞] 计算雅可比矩阵的时间导数。

```python
def M() -> numpy.ndarray
```
[非阻塞] 计算关节空间的质量矩阵。

**返回：** 对称正定质量矩阵 (n×n) [kg·m²]

```python
def C() -> numpy.ndarray
```
[非阻塞] 计算关节空间的科里奥利/离心矩阵。

**返回：** 科里奥利/离心矩阵 (n×n)

```python
def g() -> numpy.ndarray
```
[非阻塞] 计算关节空间的重力向量。

**返回：** 重力向量 (n×1) [Nm]

```python
def c() -> numpy.ndarray
```
[非阻塞] 计算关节空间的科里奥利力向量。

**返回：** 科里奥利力向量 (n×1) [Nm]

```python
def T(link_name: str) -> numpy.ndarray
```
[非阻塞] 计算指定坐标系相对于世界坐标系的变换矩阵。

**返回：** 4×4 变换矩阵

```python
def SyncURDF(template_urdf_path: str) -> None
```
[阻塞] 将连接机器人的实际运动学参数同步到模板URDF中。

```python
def reachable(pose: List[float], seed_positions: List[float], free_orientation: bool) -> Tuple[bool, List[float]]
```
[阻塞] 检查笛卡尔位姿是否可达。如果可达，还返回相应关节位置的逆运动学解。

**参数：**
- `pose`: 要检查的笛卡尔位姿
- `seed_positions`: 用作求解IK种子的关节位置
- `free_orientation`: 仅约束位置，允许方向自由移动

**返回：** (is_reachable, ik_solution) 元组

```python
def configuration_score() -> Tuple[float, float]
```
[阻塞] 获取机器人当前配置（姿态）的评分，基于可操作性测量计算。

**返回：** (translation_score, orientation_score) 元组。评分映射：差 = [0, 20), 中等 = [20, 40), 好 = [40, 100]

---

### Tool 类

管理机器人工具的接口。所有更新立即生效，无需重新上电。但机器人必须处于 IDLE 模式才能应用更改。

#### 构造函数

```python
Tool(robot: Robot)
```

#### 方法

```python
def list() -> List[str]
```
[阻塞] 获取所有已配置工具的列表。

```python
def name() -> str
```
[阻塞] 获取机器人当前使用的工具名称。如果没有活动工具，返回 "Flange"。

```python
def exist(name: str) -> bool
```
[阻塞] 检查指定工具是否已存在。

```python
def params() -> ToolParams
def params(name: str) -> ToolParams
```
[阻塞] 获取当前使用工具或指定工具的参数。

```python
def Add(name: str, params: ToolParams) -> None
```
[阻塞] 添加具有用户指定参数的新工具。

**参数：**
- `name`: 新工具名称，必须唯一
- `params`: 新工具参数

**适用模式：** IDLE

```python
def Switch(name: str) -> None
```
[阻塞] 切换到现有工具。后续所有机器人操作将默认使用此工具。

**适用模式：** IDLE

```python
def Update(name: str, params: ToolParams) -> None
```
[阻塞] 更新现有工具的参数。

**适用模式：** IDLE

```python
def Remove(name: str) -> None
```
[阻塞] 删除现有工具。不能删除 "Flange"。

**适用模式：** IDLE

```python
def CalibratePayloadParams(tool_mounted: bool) -> ToolParams
```
[阻塞] 校准工具的负载参数（质量、质心和惯量）。

**参数：**
- `tool_mounted`: 触发此校准过程时，待校准工具是否安装在机器人法兰上

**返回：** 校准后的工具参数（注意：tcp_location 将为零，应忽略）

**适用模式：** IDLE

---

## 使用示例

### 基本连接和状态读取

```python
import flexivrdk
import time

# 连接机器人
robot = flexivrdk.Robot("Rizon4s-123456")

# 等待机器人就绪
while not robot.operational():
    time.sleep(0.1)

# 读取机器人状态
states = robot.states()
print(f"关节位置: {states.q}")
print(f"TCP位姿: {states.tcp_pose}")
```

### 执行计划

```python
# 切换到计划执行模式
robot.SwitchMode(flexivrdk.Mode.NRT_PLAN_EXECUTION)

# 执行计划
robot.ExecutePlan("MyPlan")

# 等待计划完成
while robot.busy():
    time.sleep(0.1)
```

### 夹爪控制

```python
# 创建夹爪接口
gripper = flexivrdk.Gripper(robot)

# 启用夹爪
gripper.Enable("Grav")

# 移动到指定宽度
gripper.Move(width=0.05, velocity=0.1, force_limit=20)

# 抓取
gripper.Grasp(force=10)

# 读取夹爪状态
states = gripper.states()
print(f"夹爪宽度: {states.width}")
```

### 笛卡尔运动控制

```python
# 切换到笛卡尔运动力控制模式
robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)

# 获取当前TCP位姿
current_pose = robot.states().tcp_pose

# 发送新的目标位姿
target_pose = list(current_pose)
target_pose[2] += 0.1  # Z方向上移10cm

robot.SendCartesianMotionForce(target_pose)
```

---

## 注意事项

1. **实时 vs 非实时模式：** 
   - RT_* 模式需要实时循环调用（1kHz），适用于精确控制
   - NRT_* 模式使用内部运动生成器平滑命令，适用于一般应用

2. **安全：** 始终确保急停设备可用，并了解机器人的安全限制

3. **错误处理：** 大多数方法可能抛出异常，应使用 try-except 进行错误处理

4. **许可证：** Model 类需要 RDK 专业许可证

5. **线程安全：** RDK 内部使用多线程，但用户代码应避免从多个线程同时调用同一方法

---

## 版本历史

- v1.7.0: 当前版本
