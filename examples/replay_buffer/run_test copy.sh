#!/bin/bash
set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置环境变量（适配Docker环境）
export ROLL_PATH="/mnt/chensiheng/weiyu/xueban_v2/roll_dev/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# !!! 关键修复：统一时间戳，解决日志分散问题 !!!
# 生成一次时间戳，通过环境变量传递给配置文件，确保所有日志在同一目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TRAINING_TIMESTAMP="$TIMESTAMP"
OUTPUT_DIR="$SCRIPT_DIR/output/$TIMESTAMP"

# 设置wandb为离线模式
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT_DIR/wandb"

# 创建带时间戳的输出目录
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/models"
mkdir -p "$OUTPUT_DIR/tensorboard"
mkdir -p "$OUTPUT_DIR/render"
mkdir -p "$OUTPUT_DIR/wandb"

# 设置日志文件
LOG_FILE="$OUTPUT_DIR/logs/training_${TIMESTAMP}.log"

# 检测是否在tmux中运行
if [ -n "$TMUX" ]; then
    TMUX_SESSION=$(tmux display-message -p '#S')
    TMUX_WINDOW=$(tmux display-message -p '#W')
    TMUX_PANE=$(tmux display-message -p '#P')
    echo "Running in tmux session: $TMUX_SESSION, window: $TMUX_WINDOW, pane: $TMUX_PANE"

    # 在tmux中启用日志记录
    tmux pipe-pane -o "cat >> $LOG_FILE"
    echo "Tmux output will be saved to: $LOG_FILE"
    USING_TMUX=true
else
    USING_TMUX=false
fi

# 设置错误处理 - 确保tmux日志记录能正确停止
cleanup() {
    if [ "$USING_TMUX" = true ]; then
        tmux pipe-pane
        echo "Tmux logging stopped due to error/interruption"
    fi
}
trap cleanup EXIT INT TERM

echo "========================================"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (excluding GPU 0)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Training Timestamp: $TRAINING_TIMESTAMP"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "WANDB mode: offline (saved to $WANDB_DIR)"
echo "Working directory: $SCRIPT_DIR"



# 验证GPU可用性
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
else
    echo "nvidia-smi not available, skipping GPU check"
fi
echo ""

# 启动训练
cd "$ROLL_PATH"
echo "Training started at $(date)"
echo "Changing to ROLL directory: $ROLL_PATH"
echo "Using unified timestamp: $TRAINING_TIMESTAMP"

python examples/start_agentic_pipeline.py \
    --config_path ../../experiments/reproduce_frozen_lake_replay \
    --config_name agent_val_frozen_lake_replay_simple

echo "Training completed at $(date)"
echo "All logs unified in directory: $OUTPUT_DIR"
echo "Training log: $LOG_FILE"
echo "Wandb logs: $WANDB_DIR"
echo "Models saved to: $OUTPUT_DIR/models"

# 如果在tmux中，停止日志记录
if [ "$USING_TMUX" = true ]; then
    tmux pipe-pane
    echo "Tmux logging stopped"
fi

# 显示最终日志目录结构
echo ""
echo "=== Final Output Directory Structure ==="
echo "Directory: $OUTPUT_DIR"
find "$OUTPUT_DIR" -type d -name "*" | head -10
echo "========================================"
