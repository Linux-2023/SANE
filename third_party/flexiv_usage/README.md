# 非夕机械臂使用

## 整体介绍

非夕机械臂对应的库叫flexivrdk，源码在https://github.com/flexivrobotics/flexiv_rdk，有一些python的例子在`example_py`文件夹中。

本仓库里已经包含了flexivrdk的源码以及进一步整理的使用文档、测试程序和进一步封装的控制接口：
- 整理的flexivrdk的原python api在`docs/flexivrdk_python_api.md`
- 进一步封装的接口在`docs/flexiv_robot_api.md`

## 环境安装
配置T265使用环境：https://docs.qq.com/board/DVVBZU1d5YWNMYm55

配置python环境
```bash
# 创建名为 .venv 的虚拟环境
uv venv --python 3.8
# 激活环境
source .venv/bin/activate
# 安装依赖
uv pip install -r requirements.txt
```

### 碰撞检测依赖安装后处理

安装完成后，运行修复脚本解决 hppfcl 导入问题：

```bash
python collision_check/fix_hppfcl.py
```

## 配置示教器

1. 控制器开机
2. 示教器连接机器人、解除急停、开启伺服
3. 推动控制手柄上的按钮切到自动远程模式

## 开始使用

非夕机器人的api文档根据`docs/flexiv_robot_api.md`食用即可

## 双臂碰撞检测与可视化

详见 [`collision_check/README.md`](collision_check/README.md)

快速使用：
```python
from collision_check.dual_arm_collision_checker import create_checker
from collision_check.dual_arm_visualizer import DualArmVisualizer
import numpy as np

# 碰撞检测
checker = create_checker(mode="capsule")
is_collision = checker.check_collision(np.zeros(7), np.zeros(7))

# 可视化
viz = DualArmVisualizer()
viz.display(np.zeros(7), np.zeros(7))
```

交互式可视化：
```bash
python collision_check/visualize_dual_arm.py
```

## 双臂遥操作使用指南

### 1. 硬件连接与权限配置
首先确认手持夹爪已连接至上位机，并授予 USB 端口相应的读写权限（以下以 `USB0` 和 `USB1` 为例）：

```bash
# 检查设备连接状态
ls /dev/ttyUSB*

# 授予端口权限
sudo chmod 777 /dev/ttyUSB0
sudo chmod 777 /dev/ttyUSB1
```

**配置检查**：请核对 `dual_arm_config.yaml` 文件，确保 `T265相机序列号` 与 `编码器串口` 配置与实际连接一致。

### 2. 初始姿态校准
在启动程序前，需确保手持夹爪处于以下初始状态：
*   **夹爪姿态**：夹爪开口方向竖直向下，握柄水平向后。
*   **夹爪开合状态**：保持完全闭合。

### 3. 系统启动
执行以下命令启动遥操作程序。程序启动后，机械臂将自动移动至初始关节位置，然后后即可进行遥操作。

```bash
# 激活虚拟环境
source .venv/bin/activate

# 启动双臂遥操作程序
python t265_to_flexiv_mapping_dual.py
```





