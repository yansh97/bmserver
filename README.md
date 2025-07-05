# BigModel Server (BMServer)

## 概述

本项目简化大模型的权重文件管理、推理服务部署与推理性能测试。

### 当前已经实现的功能

- 本项目支持 Linux 操作系统
- 本项目支持 NVIDIA GPU 设备
- 本项目支持单节点部署张量并行（Tensor Parallel）策略
  - 如果节点中存在多个 NVIDIA GPU 设备，需要保证型号相同
- 本项目支持文本模态与多模态的对话（Chat）模型

### 未来可能实现的功能

- 本项目未来可能支持单节点部署其它并行策略
- 本项目未来可能支持文本模态或多模态的嵌入（Embedding）模型
- 本项目未来可能支持文本模态或多模态的重排序（Re-ranking）模型
- 本项目未来可能支持音频转录（Transcription）模型

### 未来不计划实现的功能

- 本项目不计划支持 Linux 以外的操作系统
- 本项目不计划支持 NVIDIA GPU 以外的设备
- 本项目不计划支持多节点部署

## 模型存储

参考 [bmhub](https://pypi.org/project/bmhub/) 中对模型存储的说明。

## 安装

```bash
pip install bmserver
```

测试 BMServer CLI 可用性。

```bash
bmserver --help
```

如果访问 Hugging Face Hub 困难，设置环境变量 `HF_ENDPOINT` 。

```bash
HF_ENDPOINT=https://hf-mirror.com bmserver --help
```

> 建议在访问 Hugging Face Hub 受限时使用 ModelScope Hub 加速下载。

## 环境变量

- `BMSERVER_POSTGRES_URL`：PostgreSQL 数据库 URL，用于存储推理性能测试结果。

## CLI 功能

### 基础功能

#### 环境检测

检测 NVIDIA GPU 设备与数量、NVIDIA 驱动版本、PyTorch 版本、Transformers 版本、vLLM 版本与自身版本。

```bash
bmserver env --help
```

### 对话（Chat）

#### 支持的模型

|        名称         |        模态        |               规模                |        量化模式         | 函数调用格式 |  推理格式   |
| :-----------------: | :----------------: | :-------------------------------: | :---------------------: | :----------: | :---------: |
|       qwen2.5       |        text        | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B | none, gptq8, gptq4, awq |    hermes    |      x      |
|    qwen2.5-coder    |        text        |   0.5B, 1.5B, 3B, 7B, 14B, 32B    | none, gptq8, gptq4, awq |    hermes    |      x      |
|         qwq         |        text        |                32B                |        none, awq        |    hermes    | deepseek-r1 |
| deepseek-r1-distill |        text        |    1.5B, 7B, 8B, 14B, 32B, 70B    |          none           |      x       | deepseek-r1 |
|     qwen2.5-vl      | text,image+,video+ |         3B, 7B, 32B, 72B          |        none, awq        |      x       |      x      |

#### 搜索支持的模型

搜索 BMServer 当前支持的对话（chat）模型，可以通过名称、规模、量化模式过滤，并查看模型是否已被下载。

默认在 Hugging Face Hub 或 ModelScope Hub 缓存目录中检索模型是否被下载。如果指定参数 `--local-dir`，则在本地模型存储目录中检索模型是否被下载。

```bash
bmserver chat search --help
```

#### 列出已下载的模型

列出已下载的对话（chat）模型，可以通过名称、规模、量化模式过滤，并查看模型占用存储空间等信息。

默认在 Hugging Face Hub 或 ModelScope Hub 缓存目录中检索已下载的模型。如果指定参数 `--local-dir`，则在本地模型存储目录中检索已下载的模型。

```bash
bmserver chat list --help
```

#### 下载模型

下载指定的对话（chat）模型，如果已下载过模型，则会更新该模型。

默认下载到 Hugging Face Hub 或 ModelScope Hub 缓存目录。如果指定参数 `--local-dir`，则下载到本地模型存储目录。

```bash
bmserver chat download --help
```

#### 更新已下载的模型

更新已下载的对话（chat）模型，可以通过名称、规模、量化模式过滤。

默认在 Hugging Face Hub 或 ModelScope Hub 缓存目录中更新已下载的模型。如果指定参数 `--local-dir`，则在本地模型存储目录中更新已下载的模型。

```bash
bmserver chat update --help
```

#### 删除已下载的模型

删除已下载的对话（chat）模型，可以通过名称、规模、量化模式过滤。

默认在 Hugging Face Hub 或 ModelScope Hub 缓存目录中更新已下载的模型。如果指定参数 `--local-dir`，则在本地模型存储目录中更新已下载的模型。

```bash
bmserver chat remove --help
```

#### 提供推理服务

在线部署指定已下载的对话（chat）模型，并提供与 OpenAI 兼容的 API 推理服务。

默认在 Hugging Face Hub 或 ModelScope Hub 缓存目录中查找已下载的模型。如果指定参数 `--local-dir`，则在本地模型存储目录中查找已下载的模型。

如果存在多个 NVIDIA GPU 设备，默认在所有 GPU 上部署模型并启动张量并行策略。如果只在部分 GPU 上部署模型，请设置 `CUDA_VISIBLE_DEVICES` 环境变量。

```bash
bmserver chat serve --help
```

#### 测试推理性能

测试指定已下载的对话（chat）模型的推理性能。支持设置多个 prompt token 数与多个 completion token 数（忽略 eos 标记），自动测试所有组合。自动探测每种组合下当前设备显存限制的最大并发数，如果最大并发数大于 256，则设置为 256 。除了最大并发数外，还会测试小于其的 2 的幂级数用于并发性能分析。

> 示例：如果探测得到的最大并发数为 9 ，则会测试如下并发数：[1, 2, 4, 8, 9]

默认在 Hugging Face Hub 或 ModelScope Hub 缓存目录中查找已下载的模型。如果指定参数 `--local-dir`，则在本地模型存储目录中查找已下载的模型。

如果存在多个 NVIDIA GPU 设备，默认在所有 GPU 上部署模型并启动张量并行策略。如果只在部分 GPU 上部署模型，请设置 `CUDA_VISIBLE_DEVICES` 环境变量。

```bash
bmserver chat benchmark --help
```
