## Piper Environment

### 一、环境构建

我们在松灵piper机械臂上构建了真机部署环境，核心组件集成在examples/piper中。由于组件主要是硬件接口，所以环境比较简单，安装方式如下：

```bash
sudo apt install can-utils 
uv venv --python 3.10 examples/piper/.venv
source examples/piper/.venv/bin/activate
uv pip install opencv-python python-can piper_sdk pyrealsense2 imageio[pyav] tyro matplotlib h5py
uv pip install -e packages/openpi-client
```

### 二、数据收集

#### 1. 硬件连接
**请在以下操作前将机械臂掰回初始状态。**
真机需要先连接硬件。首先将机械臂、realsense、全局相机的usb接口都接入pc。注意，机械臂需要给主臂和从臂接入电源，通过主臂控制机械臂，pc只负责从其中读数据。
之后通过can连接机械臂：
```bash
bash third_party/piper_sdk/piper_sdk/find_all_can_port.sh #确定是否能被can工具检测
bash third_party/piper_sdk/piper_sdk/can_activate.sh #连接
bash examples/piper/utils/can_activate_double.sh #主从臂分别使用一个can接口的连接脚本
```

先激活真机的虚拟环境
```bash
source examples/piper/.venv/bin/activate
```

连接硬件后，可通过如下代码测试硬件是否连接成功。注意，现在对于测试环境中的参数是硬编码的，可在PiperEnvironment初始化时修改相关参数，包括但不限于usb_camera_id等：
```bash
python examples/piper/test_env.py --record-mode
```

#### 2. 数据收集

运行如下代码：
```bash
python examples/piper/collect_data.py --prompt "pick up the yellow cube"
```
其中prompt为本次任务的任务指令。也可以从现有的prompt_set中选择，其内容定义在collect_data.py的开头：
```bash
python examples/piper/collect_data.py --task_type "pick" --prompt_index 0
```
运行后，程序会进入迭代循环状态（不是无限循环，注意时间），并实时可视化摄像机视角的窗口，窗口的下方显示了这次任务的prompt。此时可以进行以下三个操作
- 在窗口处按's'可进入record状态，此时窗口的字会由绿变红，可以开始遥操作执行任务；
- 在窗口处按'q'可退出record状态，此时窗口的字会由红变绿，保存本次录制的轨迹；
- 在窗口处按'esc'可退出程序；
录制的数据集会保存在./recorded_data文件夹下。


#### 3. 数据集转化
录制的格式为aloha的数据格式，我们需要将其转化为lerobot格式。转化后可以直接用来训练，并且数据会压缩到较小的体积。
我们假设数据位于./datasets/recorded_data，转化代码为：
```bash
HF_LEROBOT_HOME="./datasets/GraspAnything" uv run examples/piper/utils/convert_piper_data_to_lerobot.py --raw_dir /your/path/datasets/GraspAnything/hdf5 --repo_id piper_lerobot_data
```
其中raw_dir指的是原始数据的地址, repo_id为自定义的数据集名称。程序执行后，数据集会保存在HF_LEROBOT_HOME指向的地址，默认是/home/yourcount/.cache/huggingface/lerobot，如果想修改数据保存的地址，请修改HF_LEROBOT_HOME环境变量。


#### 4. 数据集可视化
使用rerun可视化转化后的lerobot数据集，注意datapath一定要是绝对路径，否则无法读取（huggingface的设计）。

```bash
HF_LEROBOT_HOME="./datasets/GraspAnything" uv run examples/piper/utils/vis_lerobot_datasets.py --dataset-path /home/ztlab/Project/ELM/openpi/datasets/flexiv/pickup/flexiv_lerobot_data --episode 1
```

### 三、真机推理
#### 1. 连接硬件
**请在以下操作前将机械臂掰回初始状态。**
真机需要先连接硬件。首先将机械臂、realsense、全局相机的usb接口都接入pc。注意，机械臂只能给从臂接入电源，主臂需要断电，否则无法控制机械臂。
之后通过can连接机械臂：
```bash
#确定是否能被can工具检测
bash third_party/piper_sdk/piper_sdk/find_all_can_port.sh 
#连接单个机械臂
bash third_party/piper_sdk/piper_sdk/can_activate.sh 
```
双臂可调用 [can_muti_activate.sh](third_party/piper_sdk/piper_sdk/can_muti_activate.sh)

**🔍 硬件诊断（推荐在运行前执行）**
连接硬件后，可通过如下代码测试硬件是否连接成功。注意，现在对于测试环境中的参数是硬编码的，可在PiperEnvironment初始化时修改相关参数，包括但不限于usb_camera_id, tele_mode (为true时机械臂不会动，处于安全模式，但可以读取机械臂状态)：
```bash
python examples/piper/test_env.py 
```

#### 2. 运行client
连接机械臂后，启动真机环境，代码如下，在一个terminal中运行：
```bash
source examples/piper/.venv/bin/activate #激活子虚拟环境
```
开始推理：
```bash
python examples/piper/main.py --args.prompt "pick up anything and put them in the box" --args.host "0.0.0.0" --args.high_camera_id 8 --args.left_wrist_camera_id 4
```
其中prompt参数可以更换为你想指定的prompt。host指的是server端所在设备的ip，如果要进行远端推理，可以修改ip地址。camera_id指的是相机的id，通过以下命令+插拔摄像头测试：
```bash
ls /dev/video*
sudo apt install v4l-utils
sudo v4l2-ctl --list-devices
```