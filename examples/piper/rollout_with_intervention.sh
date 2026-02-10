#!/bin/bash

# 激活虚拟环境（请根据实际情况修改路径和环境名）
cd /home/ztlab/project/ELM/recap
source examples/piper/.venv/bin/activate
# 检查虚拟环境是否激活成功
if [ $? -ne 0 ]; then
    echo "错误：虚拟环境激活失败"
    exit 1
fi

echo "虚拟环境已激活，开始执行循环任务..."

# 清理函数：终止后台进程
cleanup() {
    echo ""
    echo "检测到 Ctrl-C，正在清理..."
    
    # 终止后台进程
    echo "正在终止模型..."
    kill $pid_c 2>/dev/null
    kill $pid_d 2>/dev/null
    sleep 3
    
    if kill -0 $pid_c 2>/dev/null; then
        kill -9 $pid_c 2>/dev/null
    fi
    if kill -0 $pid_d 2>/dev/null; then
        kill -9 $pid_d 2>/dev/null
    fi
    
    exit 1
}

# 设置信号处理
trap cleanup INT

# 启动无限循环的命令c和d到后台运行[1,6](@ref)
echo "加载模型..."
uv run scripts/serve_policy.py --port 9000 policy:checkpoint --policy.config=pi05_piper_pick_cube --policy.dir=checkpoints/pi05_PourAnything/9999_torch &
pid_c=$!  # 保存命令c的进程ID

uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper_pick_cube --policy.dir=checkpoints/pi05_GraspAnything/19999_torch &
pid_d=$!  # 保存命令d的进程ID

echo "后台模型1运行中(PID: $pid_c)，模型2运行中(PID: $pid_d)"

# 等待模型加载
sleep 30

# 循环3次执行命令a和b
for i in {0..3} #..3
do
    echo "=== 第 $i 次迭代 ==="
    
    # 执行命令a
    echo "执行pour（reset）"
    python examples/piper/collect_data_with_intervention.py --prompt "Pour the objects in the box onto the table" --port 9000
    if [ $? -ne 0 ]; then
        echo "警告：pour执行时出现错误"
    fi
    
    # 休眠3秒
    sleep 3
    
    # 执行命令b
    echo "执行pick up"
    python examples/piper/collect_data_with_intervention.py --prompt "pick up anything and put them in the box" --record_mode
    if [ $? -ne 0 ]; then
        echo "警告：pickup执行时出现错误"
    fi
    
    echo "第 $i 次迭代完成"
    echo "-------------------"
done

echo "主循环任务执行完毕！"

# 终止后台命令c和d[1,2](@ref)
echo "正在终止模型..."
kill $pid_c 2>/dev/null
kill $pid_d 2>/dev/null

# 等待进程完全终止[1](@ref)
sleep 3

# 检查进程是否已终止，如果未终止则强制杀死[2](@ref)
if kill -0 $pid_c 2>/dev/null; then
    echo "模型1未正常终止，强制杀死..."
    kill -9 $pid_c
fi

if kill -0 $pid_d 2>/dev/null; then
    echo "模型2未正常终止，强制杀死..."
    kill -9 $pid_d
fi

# 再次确认进程已终止
if ! kill -0 $pid_c 2>/dev/null && ! kill -0 $pid_d 2>/dev/null; then
    echo "后台模型已成功终止"
else
    echo "警告：可能未能完全终止所有后台进程"
fi

echo "所有任务执行完毕！"