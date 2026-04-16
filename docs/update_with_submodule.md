# 更新带Submodule的ROLL代码

## 一键更新命令

在其他服务器上更新ROLL代码（包含submodule）：

```bash
# 进入项目目录
cd /path/to/ROLL

# 更新主仓库和所有submodule
git pull --recurse-submodules
```

或者分步执行：

```bash
# 1. 更新主仓库
git pull origin dev_1020

# 2. 更新submodule
git submodule update --init --recursive
```

## 完整更新脚本

创建 `update.sh`：

```bash
#!/bin/bash

# 更新主仓库
echo "更新主仓库..."
git pull origin dev_1020

# 初始化并更新submodule
echo "更新submodule..."
git submodule update --init --recursive

echo "更新完成！"
```

## 如果Submodule有问题

强制重新同步submodule：

```bash
# 完全重置submodule到远程状态
git submodule sync --recursive
git submodule update --init --recursive --force
```

## 首次克隆（新服务器）

如果是新服务器，首次克隆时就包含submodule：

```bash
git clone --recurse-submodules https://github.com/histmeisah/ROLL.git
cd ROLL
git checkout dev_1020
```

或者克隆后再初始化submodule：

```bash
git clone https://github.com/histmeisah/ROLL.git
cd ROLL
git checkout dev_1020
git submodule update --init --recursive
```

## 验证Submodule状态

```bash
# 查看submodule状态
git submodule status

# 应该显示类似：
# ec82552d2458efb486701b138c9d038e1759489c third_party/webshop-minimal (heads/main)
```

就这么简单！主要命令就是：
- `git pull --recurse-submodules` 一次更新全部
- 或者 `git pull` + `git submodule update --init --recursive` 分步更新