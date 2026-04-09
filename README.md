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
├── index.html              # 前端页面
├── main.js                 # WebRTC 逻辑 + stats 采集 + 自动模式联动
├── server.js               # 信令与 CSV 写入服务
├── auto_collect_mac.sh     # 自动切换网络环境与触发场景广播（Mac）
├── auto_fluctuate_mac.sh   # 波动网络脚本（可选）
└── real_video_csv/         # 采集到的 CSV 输出目录（运行时自动创建）
```

## 运行方式

1) 安装依赖（若已有可跳过）：
```bash
npm install
```

2) 启动服务：
```bash
node server.js
```

3) 打开浏览器（两端设备均打开）：
```
http://<服务器IP>:3000
```

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

- 输出目录：`real_video_csv/`
- 文件命名：`webrtc_network_traces_<scenario>_<traceStartTs>.csv`
- 表头字段：
```
timestamp,clientId,rtt_ms,jitter,loss_rate,recv_bps,send_bps,estimated_bw_bps
```

## 离线 RL 数据集生成（四元组）

采集到的 CSV 往往包含**两端 peer 的数据交织在同一个文件里**（用 `clientId` 区分）。

仓库内提供脚本 `tools/build_rl_dataset.py`，会对每个 CSV 做：按 `clientId` 拆分 → 按 `timestamp` 排序 → 滑动窗口聚合 → 生成离线 RL 四元组 `(s, a, r, s')`。

默认四元组定义：
- 状态 `s_t`：对最近 `window_size` 行做聚合（默认 mean），得到 `[send_bps, recv_bps, rtt_ms, loss_rate, jitter_ms]`（可用 `--state-cols` 覆盖）
- 动作 `a_t`：第 `t` 行的 `estimated_bw_bps`
- 奖励 `r_t`：QoE 形式（可调权重）：`log(1+recv_bps/1e6) - rtt_ms/1000 - loss_rate - 0.1*|a_t-a_{t-1}|/1e6`（默认用 `t+1` 行指标计算，近似动作生效后的反馈）
- 下一状态 `s_{t+1}`：窗口向后滑动 1 行

生成（输出到 `rl_dataset/`）：
```bash
python3 tools/build_rl_dataset.py --input real_video_csv --output rl_dataset --window-size 10
```s

输出文件：
- `rl_dataset/transitions.npz`：`observations/actions/rewards/next_observations/terminals`
- `rl_dataset/transitions.csv`：同内容 + `csv/scenario/trace_id/client_id/timestamp` 元信息
- `rl_dataset/manifest.csv`：每个 (csv, client_id) 的行数与样本数统计

## 自动网络脚本与权限

自动脚本需要 root 权限执行 `dnctl/pfctl`。推荐使用以下方式之一：
- 使用 `sudo node server.js` 启动服务端
- 或在系统中为 `dnctl/pfctl` 配置免密 sudo

可配置的环境变量：
- `AUTO_SCRIPT_PATH`：自动脚本路径（默认 `./auto_collect_mac.sh`）
- `SERVER_URL`：脚本通知服务端的地址（默认 `http://localhost:3000`）
- `INTERVAL_SECONDS`：切换间隔秒数（默认 `300`）

## 关键代码位置

- 本地视频生成流：`main.js` 中 `startBtn` 点击逻辑
- 发送流接入：`main.js` 中 `setupRTC()` 的 `addTrack`
- 统计采集：`main.js` 中 `startDataCollection()`
- CSV 写入：`server.js` 中 `trace_data` 事件处理

---

如需增加：音频轨、循环播放、固定码率/分辨率等能力，可继续扩展。
