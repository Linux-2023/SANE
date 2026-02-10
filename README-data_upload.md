## 1. 使用ossutil
当传输数据量超过100gb，可以通过oss进行中转，将转化的数据集上传到服务器。要在新的机器上安装ossutil，请联系@tanyz。

### 1.1 传输数据集
#### 1.1.1 数据集从本地上传到oss
我们假设数据位于./datasets文件下，可以直接上传一个文件夹，上传后本地文件夹下的内容会在新创建的云端文件夹内。上传命令如下：
```bash
ossutil cp -r ./datasets/piper_lerobot_data/ oss://umi-higroup/datasets/ELM/piper_lerobot_data/
```

#### 1.1.2 数据集从oss下载到服务器
下载文件夹，注意他是下载其中的内容，所以要接具体的数据集地址
```bash
ossutil cp -r oss://umi-higroup/datasets/ELM/piper_lerobot_data/ ./datasets/piper_lerobot_data/
```

### 1.2 传输模型
#### 1.2.1 模型从服务器上传到oss
```bash
ossutil cp -r ./checkpoints/pi05_piper_pick_cube/my_experiment/19999/ oss://umi-higroup/models/ELM/pi05_piper_pick_cube/my_experiment/19999/
```

#### 1.2.2 模型从oss下载到本地
```bash
ossutil cp -r oss://umi-higroup/models/ELM/pi05_piper_pick_cube/my_experiment/19999/ ./checkpoints/pi05_piper_pick_cube/my_experiment/19999/
```

## 2. 使用huggingface
当传输数据量小于100gb，可以通过huggingface中转。在新的机器登陆huggingface的账号，请联系@tanyz。

### 2.1 传输数据集

#### 2.1.1 创建数据集
可以通过网页端创建数据集，常用机器已登陆tanyz27账号，也可以通过命令行工具创建：
```bash
"HF_HOME=huggingface" huggingface-cli repo create papanything --repo-type dataset
```

#### 2.1.2 从本地上传到云端
```bash
"HF_HOME=huggingface" huggingface-cli upload tanyz27/papanything ./datasets/piper_lerobot_data/ --repo-type dataset
```

#### 2.1.3 从云端下载到服务器
首先先以二进制的形式保存到HF_HOME中，直接下载到指定文件夹容易卡死：
```bash
HF_HOME="huggingface" huggingface-cli download --repo-type dataset --resume-download tanyz27/papanything
```
之后运行相同的命令，但加上本地路径
```bash
HF_HOME="huggingface" huggingface-cli download --repo-type dataset --resume-download tanyz27/papanything --local-dir ./datasets/piper_lerobot_data/ --local-dir-use-symlinks False
```

### 2.2 传输模型

#### 2.2.1 创建模型
可以通过网页端创建模型，常用机器已登陆tanyz27账号，也可以通过命令行工具创建：
```bash
"HF_HOME=huggingface" huggingface-cli repo create papanything --repo-type model
```

#### 2.1.2 从服务器上传到云端
```bash
"HF_HOME=huggingface" huggingface-cli upload tanyz27/papanything ./datasets/piper_lerobot_data/ --repo-type model
```

#### 2.1.3 从云端下载到本地
首先先以二进制的形式保存到HF_HOME中，直接下载到指定文件夹容易卡死：
```bash
HF_HOME="huggingface" huggingface-cli download --repo-type model --resume-download tanyz27/papanything
```
之后运行相同的命令，但加上本地路径
```bash
HF_HOME="huggingface" huggingface-cli download --repo-type model --resume-download tanyz27/papanything --local-dir ./checkpoints/pi05_piper_pick_cube/my_experiment/19999/ --local-dir-use-symlinks False
```



