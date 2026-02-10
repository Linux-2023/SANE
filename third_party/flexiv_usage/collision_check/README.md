# 双臂碰撞检测与可视化

基于 Pinocchio + HPP-FCL 的 Flexiv Rizon 4s 双臂碰撞检测和可视化模块。

## 功能特性

- **碰撞检测**：检测双臂是否发生碰撞
- **距离计算**：计算双臂之间的最小距离
- **两种模式**：
  - `capsule`：胶囊体近似，速度快（~25万Hz），适合实时控制
  - `mesh`：精确网格，精度高，适合离线验证
- **3D 可视化**：基于 MeshCat 的浏览器可视化

## 环境安装

```bash
# 在项目根目录下
source .venv/bin/activate
uv pip install -r requirements.txt

# 修复 hppfcl 导入问题
python collision_check/fix_hppfcl.py
```

## 碰撞检测

### 基本用法

```python
from collision_check.dual_arm_collision_checker import create_checker
import numpy as np

# 创建碰撞检测器（默认使用 capsule 模式）
checker = create_checker(mode="capsule")

# 定义左右臂关节角度（7个关节，单位：弧度）
q_left = np.zeros(7)
q_right = np.zeros(7)

# 检测是否碰撞
is_collision = checker.check_collision(q_left, q_right)

# 计算最小距离（用于安全裕度）
min_dist = checker.compute_min_distance(q_left, q_right)

# 安全检查（距离 > 阈值）
is_safe = checker.is_safe(q_left, q_right, threshold=0.05)  # 5cm 安全裕度

# 获取最近点对（用于可视化）
pt_left, pt_right, dist = checker.get_closest_points(q_left, q_right)
```

### API 参考

#### `DualArmCollisionChecker`

| 方法 | 说明 |
|------|------|
| `check_collision(q_left, q_right)` | 检测是否碰撞，返回 `bool` |
| `compute_min_distance(q_left, q_right)` | 计算最小距离（米），负值表示穿透 |
| `is_safe(q_left, q_right, threshold)` | 检查距离是否大于阈值 |
| `get_closest_points(q_left, q_right)` | 返回 `(pt_left, pt_right, distance)` |
| `compute_distances(q_left, q_right)` | 返回所有碰撞对的距离列表 |
| `forward_kinematics(q_left, q_right)` | 返回双臂法兰盘的 4x4 位姿矩阵 |
| `set_mode(mode)` | 切换检测模式 (`"capsule"` 或 `"mesh"`) |

### 运行测试

```bash
python -m pytest test/test_dual_arm_collision.py -v
```

## 可视化

### 交互式可视化

```bash
python collision_check/visualize_dual_arm.py
```

启动后浏览器会自动打开 `http://127.0.0.1:7000/static/`，支持以下命令：

| 命令 | 说明 |
|------|------|
| `zero` | 零位（初始姿态） |
| `ready` | 准备姿态 |
| `reach` | 伸展姿态 |
| `close` | 靠近姿态（可能碰撞） |
| `fold` | 折叠姿态 |
| `left <j1> ... <j7>` | 设置左臂关节角（**弧度**） |
| `right <j1> ... <j7>` | 设置右臂关节角（**弧度**） |
| `both <j1> ... <j7>` | 同时设置双臂关节角（**弧度**） |
| `leftd <j1> ... <j7>` | 设置左臂关节角（**角度**） |
| `rightd <j1> ... <j7>` | 设置右臂关节角（**角度**） |
| `bothd <j1> ... <j7>` | 同时设置双臂关节角（**角度**） |
| `deg <值>` | 角度转弧度 |
| `rad <值>` | 弧度转角度 |
| `anim` | 播放演示动画 |
| `info` | 显示当前状态（含碰撞检测） |
| `help` | 查看帮助 |
| `quit` | 退出程序 |

### 代码中使用

```python
from collision_check.dual_arm_visualizer import DualArmVisualizer
import numpy as np

# 创建可视化器（自动打开浏览器）
viz = DualArmVisualizer()

# 弧度输入（默认）
q_left = np.array([0, 0.5, 0, -0.8, 0, 0.5, 0])
q_right = np.array([0, 0.5, 0, -0.8, 0, 0.5, 0])
viz.display(q_left, q_right)

# 角度输入
q_left_deg = np.array([0, 30, 0, -45, 0, 30, 0])
q_right_deg = np.array([0, 30, 0, -45, 0, 30, 0])
viz.display(q_left_deg, q_right_deg, degrees=True)

# 获取法兰盘位姿（支持 degrees 参数）
left_pose, right_pose = viz.get_flange_poses(q_left, q_right)

# 播放轨迹动画（支持 degrees 参数）
viz.display_trajectory(q_lefts_list, q_rights_list, dt=0.05, degrees=False)
```

## 文件结构

```
collision_check/
├── README.md                      # 本文档
├── dual_arm_collision_checker.py  # 碰撞检测模块
├── dual_arm_visualizer.py         # 可视化模块
└── visualize_dual_arm.py          # 交互式可视化脚本
```

## 双臂配置

双臂 URDF 文件位于 `urdf/flexiv_Rizon4s_dual_arm.urdf`：
- **左臂**：原点 `(-0.3, 0, 0)`，绕 X 轴旋转 `-90°`
- **右臂**：原点 `(0.3, 0, 0)`，绕 X 轴旋转 `-90°`
- **间距**：60cm

## 性能

| 模式 | 碰撞检测频率 | 距离计算频率 |
|------|-------------|-------------|
| capsule | ~260,000 Hz | ~38,000 Hz |
| mesh | ~200,000 Hz | ~470 Hz |

## 依赖

- `pin==2.6.21` - Pinocchio 机器人动力学库
- `hpp-fcl==2.4.4` - 碰撞检测库
- `meshcat>=0.3.2` - 3D 可视化

> **注意**：不要安装 PyPI 上的 `pinocchio` 包，它是一个无关的空包！

