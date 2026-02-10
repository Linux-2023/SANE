<<<<<<< HEAD
# noise-net
openpi-noise-net
=======
## openpi复现

### 一、环境安装
#### 1. 准备uv
openpi项目使用uv进行环境的管理，在ubuntu系统下的安装方式如下：
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. 配置openpi基础环境
##### 2.1 更新子模块
首先，使用git更新submodule：
```bash
git submodule update --init --recursive
```

##### 2.2 安装环境
之后使用uv，安装相关环境，安装方式如下：
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

##### 2.3 （optional）更改环境变量
默认设置下，模型会下载到 /home/YourCount/.cache/openpi 目录下，我们可以设置环境变量，来改变其下载地址：
```bash
export OPENPI_DATA_HOME=./checkpoints/openpi
```

#### 3. ALOHA模拟器环境构建
请参考aloha_sim环境安装的 [README](./examples/aloha_sim/README.md)

#### 4. Piper机械臂真机
请参考piper环境安装的 [README](./examples/piper/README.md)

#### 5. Flexiv非夕机械臂真机
请参考flexiv环境安装的 [README](./examples/flexiv/README.md)

---
### 二、数据收集
使用piper收集数据，请参考piper的 [README](./examples/piper/README.md)

使用flexiv收集数据，请参考flexiv的 [README](./examples/flexiv/README.md)

---
### 三、训练
#### 1. 数据集上传
请参考数据传输 [README](README-data_upload.md)，根据数据量，决策使用的工具。

#### 2. 启动训练
首先先生成数据集的norm文件。如果是6轴机械臂，可以直接使用原始的norm文件，7轴则需要根据数据集重新生成：
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_flexiv_train
```
生成后将生成的json文件放到config里的指定文件夹。

之后，可以直接通过如下代码来训练，记得在tmux中运行，以保证后台训练：
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 HF_LEROBOT_HOME="./datasets/" uv run scripts/train.py pi05_piper_pick_cube --exp-name=my_experiment --overwrite
```
其中XLA_PYTHON_CLIENT_MEM_FRACTION变量指定了gpu的用量为90%，HF_LEROBOT_HOME为数据集的地址，里面要有在训练参数中定义好的以数据集名称命名的文件夹（如piper_lerobot_data）。pi05_piper_pick_cube为训练参数，主要包括数据集的名称、训练参数、推理IO。请结合所使用的机械臂类型，使用对应的参数，参数集合定义在[config](src/openpi/training/config.py)，常用参数有：
- piper：pi05_piper_pick_cube
- flexiv (pose) pi05_flexiv_pose_train
- flexiv (qpos) pi05_flexiv_qpos_train

#### 3. 上传模型
训练完成后，模型会存储在./checkpoints/pi05_piper_pick_cube/my_experiment/19999文件夹下，19999指的是训练的step，也可以是其他的数量。该文件夹下会包含train_state, assets, params三个文件夹。如果不进行resume训练，我们只需要后两个，把train_state删掉。

上传模型的方法，请参考数据传输 [README](README-data_upload.md)，根据数据量，决策使用的工具。

---
### 四、本地推理
#### 1. 运行client端
##### 1.1 运行ALOHA_SIM client
请参考aloha_sim的 [README](./examples/aloha_sim/README.md)

##### 1.2 运行真机client
- piper：请参考piper的[README](./examples/piper/README.md)
- flexiv：请参考flexiv的[README](./examples/flexiv/README.md)


#### 2. 运行server端
之后运行server，与client进行交互。需要在一个新的terminal中运行。

##### 2.1 运行仿真server
使用pi0_aloha_sim模型在模拟器中推理
```bash
uv run scripts/serve_policy.py --env ALOHA_SIM #模拟器
```

##### 2.2 运行真机server (RTC)
###### 预训练模型
以下为预训练的调用方式，其调用的模型路径hardcode在了代码中，因此不支持使用基于torch模型的rtc

使用pi05_base模型在真机上zero-shot推理
```bash
uv run scripts/serve_policy.py --env PIPER #真机
```

###### 微调后的模型
RTC使用torch框架实现，因为其支持高效的动态图推理，支持torch推理的环境更改请参考第五章节的第0步。

首先，要将使用jax框架训练的模型转化为torch框架：
```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --config_name pi05_piper_pick_cube \
    --checkpoint_dir checkpoints/pi05_GraspandPour_Adv/19999 \
    --output_path checkpoints/pi05_GraspandPour_Adv/19999_torch
```

之后，将jax模型文件夹下的assets文件夹复制到torch模型文件夹，这其中包含了训练集的normalize参数，不可缺少。

最后，使用训练好的模型在真机上推理：
```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper_pick_cube --policy.dir=your/dir
```
其中policy.config是写死的，主要规定了数据的格式，在piper上可以复用。policy.dir为你的模型所在地址，模型的路径可以为存放torch模型的路径，也可以为jax模型路径，接口是通用的。

---

### 五、使用torch框架进行训练与推理
openpi支持使用jax和torch框架，实际测试后，对比两者的优劣如下：

##### jax框架
- 对模型加载的效率很高，可快速地启动程序，相比于torch快约10倍；
- 训练效率更高，投入相同算力，可以收敛到更低的loss；

##### torch框架
- 对于多机多卡的训练上手友好；
- 构建静态图后，推理速度更快，一个动作快10ms左右；

#### 0. 环境更改
将transformer库替换为支持torch推理的版本。首先先备份下原始的：
```bash
zip -r .venv/lib/python3.11/site-packages/transformers.zip .venv/lib/python3.11/site-packages/transformers/
```
然后将可用的transformer版本替换到虚拟环境中：
```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

#### 1. 模型转化
首先，需要将预训练模型从jax格式转化为torch格式，以下例子是转化pi05_base的命令：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --config_name pi05_piper_pick_cube \
    --checkpoint_dir checkpoints/openpi/pi05_base \
    --output_path checkpoints/openpi/pi05_base_pytorch
```

#### 2. 模型训练
torch框架的训练与原始的训练留有相同的参数。单卡训练命令如下：
```bash
uv run scripts/train_pytorch.py pi05_piper_pick_cube --exp_name exp_torch
```

多卡使用torchDDP分布式框架实现。单机多卡训练命令如下：
```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py pi05_piper_pick_cube --exp_name exp_torch --save_interval 1000
```

多机多卡训练命令如下，其中的参数设置glm平台会提供环境变量，直接使用就行：
```bash
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py pi05_piper_pick_cube --exp_name exp_torch --save_interval 1000
```
实测torchddp框架训练速度与jax框架训练速度相近。

#### 3. 模型测试

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_piper_pick_cube \
    --policy.dir=checkpoints/pi05_pickup_1113/19999
```
>>>>>>> 150a4b5 (首次上传本地文件)
