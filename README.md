# WebRTC 采集部署验证平台

本项目用于采集 WebRTC 视频通话过程中的底层网络状态数据（RTT、抖动、丢包率、带宽预估等），并进行离线强化学习（RL）模型训练与 A/B 测试验证。

***

## 完整端到端流程

```
┌─────────────────┐
│ 1. 数据集采集    │ → real_video_csv/*.csv
│ (GCC 轨迹)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. 生成训练集    │ → rl_dataset/transitions.npz
│ (四元组)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. 训练 IQL     │ → models/iql/actor.pt
│ 强化学习模型     │   models/iql/norm.json
└────────┬────────┘
         │
         ▼
┌───────────────────────────────┐
│ 4. 部署推理                   │ → HTTP/WebSocket 或 actor.onnx
│ (Python 服务 / 浏览器本地)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. A/B 测试验证  │ → ab_test_csv/*.csv
│ (GCC vs RL)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. QoE 报告     │ → output/qoe_ab_summary.csv
│ 对比分析        │
└─────────────────┘
```

***

## 项目架构

```
.
├── trace_collection/       # 数据集采集（GCC 轨迹）
│   ├── index.html
│   ├── main.js
│   └── server.js
├── ab_test/                # A/B Test（RL vs GCC）
│   ├── ab_test.html
│   ├── ab_test.js
│   ├── policy_runtime.js   # 浏览器本地 ONNX 推理
│   └── server_abtest.js
├── rl/                     # RL 训练与推理服务
│   ├── train_iql.py       # IQL 离线训练脚本
│   ├── serve_policy.py    # 推理服务（HTTP/WebSocket）
│   └── requirements.txt   # Python 依赖
├── tools/                  # 工具脚本
│   ├── build_rl_dataset.py  # 从 CSV 生成 RL 训练集
│   ├── export_iql_onnx.py   # 导出 actor.onnx
│   └── qoe_report.py        # 生成 QoE 对比报告
├── models/                 # 模型文件（训练后生成）
│   └── iql_cpu/
│       └── checkpoints/
│           └── checkpoint_50000/
│               ├── actor.pt
│               ├── actor.onnx
│               └── norm.json
├── real_video_csv/         # 数据集采集 CSV 输出
├── ab_test_csv/            # A/B Test CSV 输出
├── rl_dataset/             # RL 训练集输出
├── auto_collect_mac.sh     # 自动切换网络环境（Mac）
└── auto_fluctuate_mac.sh   # 波动网络脚本（可选）
```

***

## 详细步骤说明

### 步骤 1：数据集采集（GCC 轨迹）

采集浏览器默认 GCC 拥塞控制算法的网络轨迹数据。

#### 1.1 安装依赖

```bash
npm install
```

#### 1.2 启动采集服务（端口 3000）

```bash
# 如果需要自动切换网络，使用 sudo
sudo npm run start:collect
```

#### 1.3 浏览器打开页面

两台设备均打开：

```
http://<服务器IP>:3000
```

#### 1.4 开始采集

在每端浏览器执行：

- 选择模式（手动/自动）
- 选择本地视频文件
- 点击"加载本地视频并生成发送流"
- **仅在一侧**点击"建立连接并开始采集"

#### 输出

CSV 文件保存到：`real_video_csv/webrtc_network_traces_<scenario>_<traceStartTs>.csv`

#### 1.5 绘制原始轨迹 vs 修复后轨迹

数据集采集完成后，可以把每个 session 的原始网络轨迹与修复后的网络轨迹画成对比图，便于检查异常零点、短缺口插值和长断点切段效果。

先确保 Python 绘图依赖已安装：

```bash
python3 -m pip install -r rl/requirements.txt
```

执行绘图脚本：

```bash
python3 tools/plot_preprocessed_real_video_traces.py \
  --input real_video_csv \
  --outdir output/real_video_session_plots_compare
```

说明：

- 脚本会递归扫描 `real_video_csv/` 下的所有 CSV 文件，支持按网络场景分子目录存放的数据集
- 每个 CSV 会生成一张 `5 x 2` 对比图：左侧为原始轨迹，右侧为修复后轨迹
- 图中包含 5 组指标：`RTT`、`Jitter`、`Loss Rate`、`Throughput recv_bps`、`GCC Estimated BW`
- 修复逻辑与 `tools/build_rl_dataset.py` 保持一致：异常零点标缺失、短 gap 线性插值、长 gap 切分 `segment_id`
- 同一 `clientId` 在原始图和修复图中使用相同颜色，便于检查两个设备的对应关系

输出目录：

- `output/real_video_session_plots_compare/<scenario>/webrtc_network_traces_<scenario>_<traceStartTs>_compare.png`

***

### 步骤 2：生成 RL 训练集（四元组）

将采集到的 CSV 数据转换为强化学习训练用的四元组格式 `(s, a, r, s')`。

#### 2.1 生成训练集

```bash
python3 tools/build_rl_dataset.py \
  --input real_video_csv \
  --output rl_dataset \
  --window-size 10
```

说明：

- 脚本会递归扫描 `real_video_csv/` 下的所有 CSV 文件，支持按网络场景分子目录存放的数据集，例如 `real_video_csv/3g/*.csv`、`real_video_csv/lte/*.csv`

#### 2.2 四元组定义

- **状态 s\_t**：最近 `10` 行 `[send_bps, recv_bps, rtt_ms, loss_rate, jitter_ms]` 组成的窗口状态，按 feature-major 方式展开成 `5 x 10 = 50` 维向量，即 `[send_bps(t-9:t), recv_bps(t-9:t), rtt_ms(t-9:t), loss_rate(t-9:t), jitter_ms(t-9:t)]`
- **状态归一化**：参考 `Schaferct/code/v14_iql.py` 的处理风格，对每个 feature 使用固定尺度缩放后再截断到 `[-10, 10]`（例如 `send_bps/recv_bps` 乘 `1e-6`），并对该 feature 的 10 个时间步统一处理
- **动作 a\_t**：当前行的 `estimated_bw_bps`
- **奖励 r\_t**：QoE 形式：`log(1+recv_bps/1e6) - rtt_ms/1000 - loss_rate - 0.1*|a_t-a_{t-1}|/1e6`
- **下一状态 s\_{t+1}**：窗口向后滑动 1 行

#### 输出

- `rl_dataset/transitions.npz`：训练用的四元组数据
- `rl_dataset/transitions.csv`：带元信息的 CSV
- `rl_dataset/manifest.csv`：统计信息
- `rl_dataset/norm.json`：状态归一化参数（`state_scales` / `clip_min` / `clip_max` 等）

***

### 步骤 3：训练 IQL 强化学习模型

使用参考 `Schaferct/code/v14_iql.py` 的离线强化学习算法 IQL（Implicit Q-Learning）训练策略网络。

#### 3.1 安装 Python 依赖

```bash
python3 -m pip install -r rl/requirements.txt
```

#### 3.2 开始训练

```bash
python3 rl/train_iql.py \
  --dataset rl_dataset/transitions.npz \
  --outdir models/iql \
  --device auto
```

如果是 Apple Silicon 机器，推荐显式指定：

```bash
python3 rl/train_iql.py \
  --dataset rl_dataset/transitions.npz \
  --outdir models/iql \
  --device mps
```

使用cpu训练：

```Shell
python3 -u rl/train_iql.py --dataset rl_dataset/transitions.npz --outdir models/iql_cpu --device cpu --num-threads 4 --num-interop-threads 1 2>&1 | tee training_run_iql_cpu.log
```

#### 3.3 训练参数（可选）

- `--steps`：训练步数（默认 300000）
- `--batch`：Batch size（默认 512）
- `--vf-lr` / `--qf-lr` / `--actor-lr`：Value / Critic / Actor 学习率（默认均为 `3e-4`）
- `--iql-tau`：IQL 的 asymmetric expectile 参数（默认 `0.7`）
- `--beta`：优势加权系数（默认 3.0）
- `--discount`：折扣因子（默认 `0.99`）
- `--tau`：target Q soft update 系数（默认 `0.005`）
- `--eval-freq`：验证频率（默认 `5000`）
- `--log-freq`：日志打印频率（默认 `1000`）
- `--deterministic-actor`：切换为 deterministic actor；默认使用 Schaferct 风格的 Gaussian actor
- `--device auto`：自动选择 `cuda -> mps -> cpu`

#### 输出

- `models/iql/actor.pt`：**Actor 网络**（用于推理部署）
- `models/iql/critics.pt`：Critic 网络（仅用于训练，部署不需要）
- `models/iql/trainer.pt`：完整训练状态
- `models/iql/norm.json`：训练与状态元信息
- `models/iql/train_config.json`：本次训练使用的参数快照

#### 3.4 训练日志

建议把训练日志保存到文件，便于复盘：

```bash
python3 -u rl/train_iql.py \
  --dataset rl_dataset/transitions.npz \
  --outdir models/iql \
  --device mps \
  2>&1 | tee training_run_fixednorm.log
```

***

### 步骤 4：部署推理（两种方式）

当前项目同时支持两种部署方式：

- **方式 A：Python 推理服务（HTTP / WebSocket）**
- **方式 B：浏览器本地 ONNX 推理**

两种方式都保留，按你的场景选择：

- 如果你想快速联调、手工请求接口、单机验证：优先用 **方式 A**
- 如果你要在差网络环境下做实时限速 A/B Test：优先用 **方式 B**

#### 4.A Python 推理服务（HTTP / WebSocket）

启动推理服务，供 A/B Test 前端调用。

##### 4.A.1 安装 Python 依赖

```bash
python3 -m pip install -r rl/requirements.txt
```

当前 `rl/serve_policy.py` 已与训练产物对齐，能够：

- 读取 `norm.json` 中的 `window_size`、`state_dim`、`state_layout`、`actor_type` 等元信息
- 维护最近 `10` 步状态窗口，并按 feature-major 方式展开成 `5 x 10 = 50` 维输入
- 使用训练导出的 `actor.pt` 进行在线推理
- 对 HTTP 和 WebSocket 分别维护推理状态窗口
- 保留动作平滑、限幅和 fallback 保护逻辑

##### 4.A.2 启动服务（端口 8000）

如果你使用旧目录结构：

```bash
python3 rl/serve_policy.py \
  --model models/iql/actor.pt \
  --norm models/iql/norm.json \
  --port 8000
```

如果你使用新的 checkpoint 目录结构，例如 `checkpoint_50000`：

```bash
python3 rl/serve_policy.py \
  --model models/iql_cpu/checkpoints/checkpoint_50000/actor.pt \
  --norm models/iql_cpu/checkpoints/checkpoint_50000/norm.json \
  --port 8000
```

##### 4.A.3 HTTP 接口

```text
POST http://<host>:8000/predict
Content-Type: application/json

{
  "state": {
    "send_bps": 1000000,
    "recv_bps": 1000000,
    "rtt_ms": 50,
    "loss_rate": 0.01,
    "jitter_ms": 5
  },
  "prev_action_bps": 2000000,
  "fallback_action_bps": 2000000
}
```

响应示例：

```json
{
  "action_bps": 2500000,
  "raw_action_bps": 2600000,
  "clipped": false,
  "smoothed": true,
  "fallback_used": false
}
```

说明：

- 每次请求只需要传入**当前一步**的 `state`
- 服务端会自动把当前状态追加到内部历史窗口，并补齐到最近 `10` 步后再送入模型
- HTTP 方式使用进程级共享窗口，适合单路调试或串行请求场景
- 如果切换到新的会话/新的客户端，建议重启服务，避免沿用上一段会话的历史状态

##### 4.A.4 WebSocket 接口

```text
ws://<host>:8000/ws
```

说明：

- WebSocket 连接建立后，服务端会为该连接单独维护一份 `10` 步历史窗口
- 同一连接内连续发送当前一步 `state`，后端会自动滚动更新窗口并推理
- 多个 WebSocket 客户端之间的历史状态互不影响，更适合在线多客户端场景

##### 4.A.5 推理输入输出约定

- 输入状态字段固定为：`send_bps`、`recv_bps`、`rtt_ms`、`loss_rate`、`jitter_ms`
- 模型真实接收的是服务端维护后的 `50` 维窗口状态：
  `[send_bps(t-9:t), recv_bps(t-9:t), rtt_ms(t-9:t), loss_rate(t-9:t), jitter_ms(t-9:t)]`
- 当历史不足 `10` 步时，服务端会使用当前已知最早的一帧做左侧补齐
- 返回值中：
  - `action_bps` 为经过裁剪与平滑后的最终限速值
  - `raw_action_bps` 为模型原始输出映射回 bps 后的结果
  - `fallback_used` 表示本次是否触发了 fallback 保护

#### 4.B 浏览器本地 ONNX 推理

推荐在实时 A/B Test 中使用这种方式，把训练好的 Actor 导出为 `ONNX`，然后让每台设备在浏览器端本地推理。

这样做的好处：

- 不需要另一台设备上的远端推理服务
- 不依赖局域网 RTT，差网络环境下也能实时决策
- 每个浏览器实例本地维护自己的 `10` 步滑动窗口，不会跨设备串扰
- 更适合当前 WebRTC 限速这种强实时控制环路

##### 4.B.1 安装前端依赖

在项目根目录执行：

```bash
npm install
```

说明：

- 这里会安装 `onnxruntime-web`
- `ab_test/server_abtest.js` 会自动把 `models/` 和 `onnxruntime-web` 的浏览器运行时脚本作为静态资源暴露出来

##### 4.B.2 导出 ONNX

训练完成后，把某个 checkpoint 的 `actor.pt` 导出为 `actor.onnx`。

示例：

```bash
python3 tools/export_iql_onnx.py \
  --model models/iql_cpu/checkpoints/checkpoint_50000/actor.pt \
  --norm models/iql_cpu/checkpoints/checkpoint_50000/norm.json \
  --out models/iql_cpu/checkpoints/checkpoint_50000/actor.onnx
```

说明：

- 导出脚本参考 `Schaferct/code/v14_iql.py` 的 `export2onnx()` 思路实现
- 默认会对导出的 ONNX 做一次 PyTorch/ONNX 输出一致性校验
- 如果只是快速导出，可额外加 `--skip-verify`

##### 4.B.3 浏览器本地推理的输入输出

前端本地推理时，浏览器会：

- 从 WebRTC stats 读取当前一步状态：
  - `send_bps`
  - `recv_bps`
  - `rtt_ms`
  - `loss_rate`
  - `jitter_ms`
- 在浏览器端本地维护最近 `10` 步历史窗口
- 在归一化之前先做一层**轻量在线清洗**
  - 对 `send_bps`、`recv_bps`、`rtt_ms`、`jitter_ms` 做非负裁剪
  - 对 `loss_rate` 裁剪到 `[0, 1]`
  - 对可疑异常零点做抑制，优先用最近有效值/最近短窗口统计值回填
  - 当连续异常超过短 gap 阈值时，重置本地历史窗口，避免长断点污染状态
- 再按 `norm.json` 中的配置做固定尺度缩放、截断、展开
- 本地加载 `actor.onnx` 并推理出动作
- 在本地调用 `RTCRtpSender.setParameters()` 设置 `maxBitrate`

控制闭环变成：

```text
本地 WebRTC stats -> 在线清洗 -> 本地滑动窗口 -> 本地 ONNX 推理 -> 本地设置 maxBitrate
```

整个过程不经过局域网推理请求。

补充说明：

- 这层在线清洗位于 `ab_test/policy_runtime.js`
- 它的目标是尽量缓解训练阶段 `clean_df()` 与部署阶段实时统计之间的分布差异
- 它是**因果、轻量、实时版**近似处理，并不是对 `tools/build_rl_dataset.py` 中离线 `clean_df()` 的完全复刻
- 当前实现优先解决最常见的异常零点问题，尽量在不增加明显浏览器端延迟的前提下提高推理稳定性

***

### 步骤 5：A/B 测试验证（GCC vs RL）

对比 GCC 和 RL 限速两种拥塞控制算法的 QoE 差异。

#### 5.1 先确定你要使用哪种推理方式

- 如果使用 **Python 推理服务**：
  - 先按 `4.A` 启动 `rl/serve_policy.py`
- 如果使用 **浏览器本地 ONNX 推理**：
  - 先按 `4.B` 导出 `actor.onnx`
  - 确认目标 checkpoint 目录下已经有 `actor.onnx` 和 `norm.json`

#### 5.2 启动 A/B Test 服务（端口 3001）

```bash
# 如果需要自动切换网络，使用 sudo
sudo env SERVER_URL=http://localhost:3001 npm run start:ab
```

#### 5.3 浏览器打开页面

两台设备均打开：

```
http://<服务器IP>:3001/ab_test.html
```

#### 5.4 配置 AB 分组

在页面中选择分组模式：

- **对照组：GCC**：仅使用浏览器 GCC
- **实验组：RL + 限速**：使用 RL 模型决策
- **随机 (50/50)**：每次连接随机分配（推荐用于 A/B 测试）

#### 5.5 选择推理方式

页面里现在支持两种推理方式：

- **浏览器本地 ONNX**：推荐，默认选项
- **远端 HTTP 服务**：兼容旧方案

推荐使用浏览器本地 ONNX 时：

- `推理方式` 选择 `浏览器本地 ONNX`
- `本地模型` 填写：
  - `/models/iql_cpu/checkpoints/checkpoint_50000/actor.onnx`
- `归一化` 填写：
  - `/models/iql_cpu/checkpoints/checkpoint_50000/norm.json`

默认页面已经预填了上面这组路径。

如果使用远端 HTTP 服务：

- `推理方式` 选择 `远端 HTTP 服务`
- `推理服务` 填写：
  - `http://<部署 serve_policy.py 的设备 IP>:8000`

#### 5.6 开始采集

- 选择模式（手动/自动）
- 选择本地视频文件
- 点击"加载本地视频并生成发送流"
- **仅在一侧**点击"建立连接并开始采集"

#### 输出

CSV 文件会根据推理方式写入不同目录：

- 如果页面选择 **浏览器本地 ONNX**：
  - `ab_test_onnx_csv/webrtc_abtest_traces_<scenario>_<traceStartTs>.csv`
- 如果页面选择 **远端 HTTP 服务**：
  - `ab_test_csv/webrtc_abtest_traces_<scenario>_<traceStartTs>.csv`

说明：

- 两种推理方式使用相同的文件命名规则，区别只在输出目录
- 这样可以把“浏览器本地推理实验结果”和“远端服务推理实验结果”分开保存，方便后续做 QoE 对比和问题排查

***

### 步骤 6：生成 QoE 对比报告

分析 A/B Test 数据，生成 GCC vs RL 的 QoE 对比报告。

#### 6.1 生成报告

```bash
python3 tools/qoe_report.py \
  --input ab_test_csv \
  --outdir output
```

#### 6.2 QoE 指标

报告包含以下指标：

- `recv_bps_mean`：平均接收带宽（吞吐）
- `rtt_ms_p95`：95 分位 RTT（时延）
- `loss_rate_mean`：平均丢包率
- `jitter_ms_p95`：95 分位抖动
- `cap_delta_bps_mean`：码率变化平滑性
- `qoe_score_mean`：综合 QoE 分数

#### 6.3 输出文件

- `output/qoe_segments.csv`：每条 trace 按 client 统计的 QoE 指标
- `output/qoe_ab_summary.csv`：按 `(scenario, ab_group)` 聚合的 A/B 对比表

#### 6.4 显著性检验（p 值）

如果你希望判断某个 QoE 指标在 GCC vs RL 之间的差异是否可能由随机波动造成，可以基于 `output/qoe_segments.csv` 做显著性检验。

本仓库提供 `tools/qoe_significance.py`，默认对每个 `scenario`、每个指标执行 **Welch t-test（双侧）**，并输出 p 值；也支持更保守的 trace 级别聚合检验。

```bash
# 1) 先生成 QoE 报告（得到 qoe_segments.csv）
python3 tools/qoe_report.py --input ab_test_csv --outdir output

# 2) 计算显著性检验（p 值）
python3 tools/qoe_significance.py --segments output/qoe_segments.csv --outdir output

# 可选：更保守的 trace 级别检验（建议论文/报告中使用）
python3 tools/qoe_significance.py --segments output/qoe_segments.csv --outdir output --unit trace
```

输出文件：

- `output/qoe_significance.csv`：每个 `(scenario, metric)` 的样本量、均值差、t 统计量、自由度与 p 值；默认同时给出 BH-FDR 校正后的 `p_value_bh` 以便多重比较。

***

## 平台两种功能（完全隔离）

本仓库同时支持两条完全隔离的链路：

| 功能           | 输出目录                | 用途                            |
| ------------ | ------------------- | ----------------------------- |
| **数据集采集**    | `real_video_csv/`   | 用于离线 RL 数据集构建与训练              |
| **A/B Test** | `ab_test_csv/`      | 使用远端 HTTP 推理服务时的在线验证与 QoE 对比  |
| **A/B Test** | `ab_test_onnx_csv/` | 使用浏览器本地 ONNX 推理时的在线验证与 QoE 对比 |

> 建议：两条链路用不同端口启动不同 Node 服务，避免误写同目录/误读同 CSV。

***

## 自动网络脚本与权限

自动脚本需要 root 权限执行 `dnctl/pfctl`。推荐使用以下方式之一：

- 使用 `sudo npm run start:collect` 或 `sudo npm run start:ab` 启动服务端
- 或在系统中为 `dnctl/pfctl` 配置免密 sudo

可配置的环境变量：

- `AUTO_SCRIPT_PATH`：自动脚本路径（默认 `./auto_collect_mac.sh`）
- `SERVER_URL`：脚本通知服务端的地址（数据集采集默认 `http://localhost:3000`，A/B Test 默认 `http://localhost:3001`）
- `INTERVAL_SECONDS`：切换间隔秒数（默认 `300`）

***

## 关键代码位置

### 数据集采集（GCC 轨迹）

- 本地视频生成流：`trace_collection/main.js` 中 `startBtn` 点击逻辑
- 发送流接入：`trace_collection/main.js` 中 `setupRTC()` 的 `addTrack`
- 统计采集：`trace_collection/main.js` 中 `startDataCollection()`
- CSV 写入：`trace_collection/server.js` 中 `trace_data` 事件处理

### A/B Test（RL vs GCC）

- AB 分组逻辑：`ab_test/ab_test.js`
- 浏览器本地推理与在线清洗：`ab_test/policy_runtime.js`
- 前端限速控制：`ab_test/ab_test.js`
- CSV 写入：`ab_test/server_abtest.js` 中 `trace_data` 事件处理

### RL 训练与推理

- IQL 训练：`rl/train_iql.py`
- ONNX 导出：`tools/export_iql_onnx.py`
- 浏览器本地推理运行时：`ab_test/policy_runtime.js`
- Python 推理服务：`rl/serve_policy.py`
- 数据集生成：`tools/build_rl_dataset.py`
- QoE 报告：`tools/qoe_report.py`

***

## Actor vs Critic 模型说明

| 模型         | 用途                   | 保存文件         | 部署时是否需要   |
| ---------- | -------------------- | ------------ | --------- |
| **Actor**  | 实际用于决策的策略网络，输入状态输出动作 | `actor.pt`   | ✅ **需要**  |
| **Critic** | 训练时用于评估动作价值（Q1/Q2/V） | `critics.pt` | ❌ **不需要** |

***

如需增加：音频轨、循环播放、固定码率/分辨率等能力，可继续扩展。
