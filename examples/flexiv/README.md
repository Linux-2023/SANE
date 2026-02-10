## Flexiv Environment

### 一、环境构建

我们在非夕 Flexiv 机械臂上构建了真机部署环境，核心组件集成在 examples/flexiv 中。由于 flexivrdk 需要 Python 3.8 环境，安装方式如下：

#### 1 配置 Python 环境
```bash
# 创建 Python 3.8 虚拟环境
uv venv --python 3.8 examples/flexiv/.venv
source examples/flexiv/.venv/bin/activate

# 安装依赖
uv pip install -r third_party/flexiv_usage/requirements.txt
uv pip install opencv-python imageio[pyav] tyro matplotlib h5py
uv pip install -e packages/openpi-client
```

测试：
```bash
source examples/flexiv/.venv/bin/activate
python examples/flexiv/env.py --robot_sn Rizon4s-063239 --camera_ids 0 --tele_mode
```
其中 `--tele_mode` 表示不通过env来控制机器人运动；`--robot_sn` 需要替换为你的机器人序列号。

后面部分为遥操作

#### 2 碰撞检测依赖后处理
安装完成后，运行修复脚本解决 hppfcl 导入问题：
```bash
python third_party/flexiv_usage/collision_check/fix_hppfcl.py
```

#### 3 配置示教器
1. 控制器开机
2. 示教器连接机器人、解除急停、开启伺服
3. 推动控制手柄上的按钮切到自动远程模式

#### 4 （可选）配置 T265 相机用于遥操作
如果需要使用 T265 进行遥操作数据采集，请参考：https://docs.qq.com/board/DVVBZU1d5YWNMYm55

遥操作时需要确认手持夹爪已连接，并授予 USB 端口权限：
```bash
# 检查设备连接状态
ls /dev/ttyUSB*

# 授予端口权限
sudo chmod 777 /dev/ttyUSB0
sudo chmod 777 /dev/ttyUSB1
```

### 二、数据收集

#### 1. 硬件连接
将非夕机械臂的网线、所有摄像头，以及t265手爪都接入PC。之后给t265的端口授予权限（参考环境构建部份）。

#### 2. 数据收集
```bash
python examples/flexiv/collect_data.py --prompt "pick up the yellow cube"
```
其中prompt为本次任务的任务指令。运行后，程序会进入迭代循环状态（不是无限循环，注意时间），并实时可视化摄像机视角的窗口，窗口的下方显示了这次任务的prompt。此时可以进行以下三个操作
- 在窗口处按's'可进入record状态，此时窗口的字会由绿变红，可以开始遥操作执行任务；
- 在窗口处按'q'可退出record状态，此时窗口的字会由红变绿，保存本次录制的轨迹；
- 在窗口处按'esc'可退出程序；
录制的数据集会保存在./recorded_data文件夹下，其他可修改参数请参考collect_data.py的代码部份。

#### 3. 数据集转化
录制的格式为hdf5的数据格式，我们需要将其转化为lerobot格式。转化后可以直接用来训练，并且数据会压缩到较小的体积。
我们假设数据位于./datasets/recorded_data，转化代码为：
```bash
HF_LEROBOT_HOME="./datasets/GraspAnything" uv run examples/flexiv/utils/convert_flexiv_data_to_lerobot.py --raw_dir /your/path/datasets/GraspAnything/hdf5 --repo_id flexiv_lerobot_data --state_type pose
```
其中raw_dir指的是原始数据的地址, repo_id为自定义的数据集名称。程序执行后，数据集会保存在HF_LEROBOT_HOME指向的地址，默认是/home/yourcount/.cache/huggingface/lerobot。如果想修改数据保存的地址，请修改HF_LEROBOT_HOME环境变量。

#### 4. 数据集可视化
使用rerun可视化转化后的lerobot数据集，注意datapath一定要是绝对路径，否则无法读取（huggingface的设计）。

```bash
HF_LEROBOT_HOME="./datasets/GraspAnything" uv run examples/flexiv/utils/vis_lerobot_datasets.py --dataset-path /home/ztlab/Project/ELM/openpi/datasets/flexiv/pickup/flexiv_lerobot_data --episode 1
```

### 三、真机推理
#### 1. 连接硬件
真机需要先连接硬件。首先将机械臂、相机的usb接口都接入pc。在示教器中将机械臂切换为远程命令模式后，通过如下命令测试：
```bash
source examples/flexiv/.venv/bin/activate
python examples/flexiv/env.py --robot_sn Rizon4s-063239 --camera_ids 0 --tele_mode
```
其中 `--tele_mode` 表示不通过env来控制机器人运动；`--robot_sn` 需要替换为你的机器人序列号。

#### 2. 运行client
连接机械臂后，启动真机环境，代码如下，在一个terminal中运行：
```bash
source examples/flexiv/.venv/bin/activate #激活子虚拟环境
```
开始推理：
```bash
python examples/flexiv/main.py --args.prompt "pick up anything and put them in the box" --args.host "0.0.0.0"
```
其中prompt参数可以更换为你想指定的prompt。host指的是server端所在设备的ip，如果要进行远端推理，可以修改ip地址。硬件相关参数定义在[dual_arm_env_config.yaml](examples/flexiv/dual_arm_env_config.yaml)。camera_id指的是相机的id，通过以下命令+插拔摄像头测试：
```bash
ls /dev/video*
sudo apt install v4l-utils
sudo v4l2-ctl --list-devices
```