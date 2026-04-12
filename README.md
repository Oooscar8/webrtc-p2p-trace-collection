# WebRTC 强化学习网络轨迹采集终端（真实视频流版）

本项目用于采集 WebRTC 视频通话过程中的底层网络状态数据（RTT、抖动、丢包率、带宽预估等），供后续强化学习（RL）模型训练与分析。当前版本使用本地真实视频文件作为发送源，更接近真实 RTC 场景，并支持手动/自动切换网络环境与按段落盘。

## 项目架构

项目采用 Client-Server（C/S）架构辅助建立 Peer-to-Peer（P2P）通信：
- **服务端 (Node.js + Socket.IO)**：
  - 提供静态页面托管
  - 作为 WebRTC 信令服务器（转发 SDP/ICE）
  - 接收前端上报的网络统计并写入 CSV 文件
  - 在自动模式下可启动本机网络环境切换脚本，并广播网络场景变化
- **客户端 (HTML5 + WebRTC API)**：
  - 选择本地视频文件并生成可发送的 `MediaStream`
  - 建立 WebRTC P2P 连接
  - 使用 `getStats()` 定时采集网络指标并上报
  - 自动模式下接收场景广播并切换 trace 段落

## 目录结构

```
.
├── trace_collection/       # 数据集采集（GCC 轨迹）
│   ├── index.html
│   ├── main.js
│   └── server.js
├── ab_test/                # A/B Test（RL vs GCC）
│   ├── ab_test.html
│   ├── ab_test.js
│   └── server_abtest.js
├── rl/                     # RL 训练与推理服务
├── tools/                  # 工具脚本（数据集生成、QoE 报告）
├── models/                 # 模型文件
├── real_video_csv/         # 数据集采集 CSV 输出
├── ab_test_csv/            # A/B Test CSV 输出
├── auto_collect_mac.sh     # 自动切换网络环境与触发场景广播（Mac）
└── auto_fluctuate_mac.sh   # 波动网络脚本（可选）
```

## 运行方式

### 数据集采集（GCC 轨迹）

1) 安装依赖（若已有可跳过）：
```bash
npm install
```

2) 启动服务：
```bash
npm run start:collect
```

3) 打开浏览器（两端设备均打开）：
```
http://<服务器IP>:3000
```

### A/B Test（RL vs GCC）

参考下方 "平台两种功能" 章节

4) 在每端浏览器执行以下步骤：
   - 选择模式（手动/自动）
   - 选择本地视频文件
   - 点击“加载本地视频并生成发送流”
   - 仅在一侧点击“建立连接并开始采集”

## 两端部署与采集流程

### 手动模式
- 两端都选择手动模式
- 仅在一侧点击“建立连接并开始采集”
- 通过前端下拉框手动选择场景，CSV 会按场景分段落盘

### 自动模式（推荐）
- 两端都选择自动模式
- 仅在一侧点击“建立连接并开始采集”（发起端）
- 服务端收到 offer 时会启动本机脚本 `auto_collect_mac.sh`
- 脚本每 5 分钟切换一次网络环境，并通过 `/auto/network` 广播给两端
- 两端前端会同步显示当前场景，并自动切分 trace 段落盘

说明：脚本只会在运行 `server.js` 的那台机器上执行，所以可以是 Windows+Mac，也可以是双 Mac，只需保证发起端通过服务端触发即可。

## P2P 通信流程简述

1. **信令交互**：通过 Socket.IO 交换 Offer/Answer
2. **ICE 交换**：通过 STUN 获取候选地址并互换
3. **媒体传输**：本地视频 `captureStream()` 生成 `MediaStream`，通过 `RTCPeerConnection.addTrack()` 发送
4. **对端渲染**：对端在 `ontrack` 中将流挂载到 `remoteVideo`

## CSV 数据输出

### 数据集采集（GCC 轨迹）

- 输出目录：`real_video_csv/`
- 文件命名：`webrtc_network_traces_<scenario>_<traceStartTs>.csv`
- 表头字段：
```
timestamp,clientId,rtt_ms,jitter,loss_rate,recv_bps,send_bps,estimated_bw_bps
```

### A/B Test（RL vs GCC）

- 输出目录：`ab_test_csv/`
- 文件命名：`webrtc_abtest_traces_<scenario>_<traceStartTs>.csv`
- 表头字段：
```
timestamp,clientId,ab_group,rtt_ms,jitter,loss_rate,recv_bps,send_bps,gcc_estimated_bw_bps,policy_max_bitrate_bps,estimated_bw_bps
```
字段说明：
- `ab_group`：`gcc`(对照组) / `rl`(实验组)
- `gcc_estimated_bw_bps`：浏览器 `getStats()` 中的 `availableOutgoingBitrate`
- `policy_max_bitrate_bps`：实验组实际下发给 sender 的 `maxBitrate`（对照组为 0）
- `estimated_bw_bps`：用于对比/训练的“动作”字段；对照组=GCC 预估，实验组=RL 限速值

## 离线 RL 数据集生成（四元组）

采集到的 CSV 往往包含**两端 peer 的数据交织在同一个文件里**（用 `clientId` 区分）。

仓库内提供脚本 `tools/build_rl_dataset.py`，会对每个 CSV 做：按 `clientId` 拆分 → 按 `timestamp` 排序 → 滑动窗口聚合 → 生成离线 RL 四元组 `(s, a, r, s')`。

默认四元组定义：
- 状态 `s_t`：对最近 `window_size` 行做聚合（默认 mean），得到 `[send_bps, recv_bps, rtt_ms, loss_rate, jitter_ms]`（可用 `--state-cols` 覆盖；其中 `jitter_ms` 对应 CSV 列 `jitter`）
- 动作 `a_t`：第 `t` 行的 `estimated_bw_bps`
- 奖励 `r_t`：QoE 形式（可调权重）：`log(1+recv_bps/1e6) - rtt_ms/1000 - loss_rate - 0.1*|a_t-a_{t-1}|/1e6`（默认用 `t+1` 行指标计算，近似动作生效后的反馈）
- 下一状态 `s_{t+1}`：窗口向后滑动 1 行

生成（输出到 `rl_dataset/`）：
```bash
python3 tools/build_rl_dataset.py --input real_video_csv --output rl_dataset --window-size 10
```

输出文件：
- `rl_dataset/transitions.npz`：`observations/actions/rewards/next_observations/terminals`
- `rl_dataset/transitions.csv`：同内容 + `csv/scenario/trace_id/client_id/timestamp` 元信息
- `rl_dataset/manifest.csv`：每个 (csv, client_id) 的行数与样本数统计

## 离线训练（IQL baseline）

> 说明：当前数据集来自 GCC 行为策略的离线轨迹。IQL 更偏向“数据内”的策略提升，适合作为第一版离线 RL baseline。

1) 安装 Python 依赖：
```bash
python3 -m pip install -r rl/requirements.txt
```

2) 训练（会输出 `models/iql/actor.pt` 与 `models/iql/norm.json`）：
```bash
python3 rl/train_iql.py --dataset rl_dataset/transitions.npz --outdir models/iql
```

## 推理服务（HTTP / WebSocket）

启动推理服务：
```bash
python3 rl/serve_policy.py --model models/iql/actor.pt --norm models/iql/norm.json --port 8000
```

- HTTP: `POST http://<host>:8000/predict`
- WS: `ws://<host>:8000/ws`

## 平台两种功能（完全隔离）

本仓库同时支持两条完全隔离的链路：
- **数据集采集（GCC 轨迹）**：用于离线 RL 数据集构建与训练，输出到 `real_video_csv/`，字段与历史数据保持一致。
- **A/B Test（RL vs GCC）**：用于部署模型在线验证与 QoE 对比，输出到 `ab_test_csv/`，字段包含 AB 分组与限速信息。

> 建议：两条链路用不同端口启动不同 Node 服务，避免误写同目录/误读同 CSV。

### 数据集采集（GCC 轨迹）
启动采集服务（端口 3000）：
```bash
npm install
npm run start:collect
```
浏览器打开：`http://<服务器IP>:3000`（页面：`index.html`）

### A/B Test（RL vs GCC）
1) 启动推理服务（端口 8000）：

先确认本地存在模型文件（训练后会生成）：
- `models/iql/actor.pt`
- `models/iql/norm.json`

```bash
python3 -m pip install -r rl/requirements.txt
python3 rl/serve_policy.py --model models/iql/actor.pt --norm models/iql/norm.json --port 8000
```

> 如果你当前 `serve_policy.py` 版本已提供默认值，也可以只传 `--port 8000`；若出现“--model/--norm required”，请按上面显式传参。

2) 启动 A/B Test 服务（端口 3001）：
```bash
npm install
npm run start:ab
```

3) 浏览器打开：`http://<服务器IP>:3001/ab_test.html`，在“AB 分组”里选择 `对照组：GCC` / `实验组：RL + 限速` / `随机(50/50)`。

说明：A/B Test 前端对推理请求做了限频（默认 1s 1 次），并将返回的 `action_bps` 写入 `maxBitrate`；同时会在 CSV 中记录 `gcc_estimated_bw_bps` 与 `policy_max_bitrate_bps`，用于 QoE 对比。

## A/B 对比与 QoE 指标（离线统计）

为了对比 RL 拥塞控制 vs GCC，建议至少输出以下 QoE 指标组（`tools/qoe_report.py` 已覆盖）：吞吐（`recv_bps_mean`）、时延（`rtt_ms_p95`）、丢包（`loss_rate_mean`）、抖动（`jitter_ms_p95`）、平滑性（`cap_delta_bps_mean`）、综合 QoE（`qoe_score_mean`）。

生成 A/B QoE 报告：
```bash
python3 tools/qoe_report.py --input ab_test_csv --outdir output
```

输出：
- `output/qoe_segments.csv`：每条 trace 按 client 统计的 QoE 指标
- `output/qoe_ab_summary.csv`：按 (scenario, ab_group) 聚合后的 A/B 对比表

## 自动网络脚本与权限

自动脚本需要 root 权限执行 `dnctl/pfctl`。推荐使用以下方式之一：
- 使用 `sudo npm run start:collect` 或 `sudo npm run start:ab` 启动服务端
- 或在系统中为 `dnctl/pfctl` 配置免密 sudo

可配置的环境变量：
- `AUTO_SCRIPT_PATH`：自动脚本路径（默认 `./auto_collect_mac.sh`）
- `SERVER_URL`：脚本通知服务端的地址（数据集采集默认 `http://localhost:3000`，A/B Test 默认 `http://localhost:3001`）
- `INTERVAL_SECONDS`：切换间隔秒数（默认 `300`）

## 关键代码位置

### 数据集采集（GCC 轨迹）
- 本地视频生成流：`trace_collection/main.js` 中 `startBtn` 点击逻辑
- 发送流接入：`trace_collection/main.js` 中 `setupRTC()` 的 `addTrack`
- 统计采集：`trace_collection/main.js` 中 `startDataCollection()`
- CSV 写入：`trace_collection/server.js` 中 `trace_data` 事件处理

### A/B Test（RL vs GCC）
- AB 分组逻辑：`ab_test/ab_test.js`
- RL 策略请求：`ab_test/ab_test.js`
- CSV 写入：`ab_test/server_abtest.js` 中 `trace_data` 事件处理

---

如需增加：音频轨、循环播放、固定码率/分辨率等能力，可继续扩展。
