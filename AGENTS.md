# 项目开发指南

## 环境

- 项目命令统一使用已有的 `py39` Conda 环境，例如
  `conda run -n py39 ...`，不需要先检查环境，也不要重复创建环境。
- 使用仓库 `.env` 中配置的 Hugging Face 和 Docker 镜像源。

## 长时间运行的命令

- 所有执行时间较长的命令都必须在命名的 `tmux` 会话中运行，包括但不限于
  依赖安装、模型下载、Docker 拉取或构建，以及推理和评测命令，避免网络
  波动或终端断开导致任务中止。
- 示例：
  先运行 `tmux new -s testgeneval-install`，再执行目标命令，最后按
  `Ctrl-b d` 保持会话在后台运行。
- 使用 `tmux attach -t testgeneval-install` 重新连接，并在开始新的安装或
  评测前检查任务进度。

## 构建与测试

- 按照 [README.md](README.md) 中的命令运行流水线和评测。
- Docker、PyPI、Conda 和 Hugging Face 下载应优先使用已配置的国内镜像源。
- GitHub 仓库和文件下载使用 `https://gitclone.com` 加速地址。
- GitHub 仓库下载按清华 Git 镜像、`bgithub.xyz`、`gitclone.com`、`github.ur1.fun`、
  `gitee.com/mirrors`、官方 GitHub 的顺序自动回退；Gitee 仓库名称不一致时
  继续尝试后续地址。
- Docker 构建使用串行方式执行；任一步骤失败后立即停止，不等待其他并行任务。

- GitHub 仓库下载优先使用 `https://github.ur1.fun`，如不可用再切换其他可用代理。
