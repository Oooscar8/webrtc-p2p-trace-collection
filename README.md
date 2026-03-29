# WebRTC 强化学习网络轨迹采集终端（真实视频流版）

本项目用于采集 WebRTC 视频通话过程中的底层网络状态数据（RTT、抖动、丢包率、带宽预估等），供后续强化学习（RL）模型训练与分析。当前版本使用**本地真实视频文件**作为发送源，较原先伪造视频流更接近真实 RTC 场景。

## 项目架构

项目采用 **Client-Server（C/S）** 架构辅助建立 **Peer-to-Peer（P2P）** 通信：
- **服务端 (Node.js + Socket.IO)**：
  - 提供静态页面托管
  - 作为 WebRTC 信令服务器（转发 SDP/ICE）
  - 接收前端上报的网络统计并写入 CSV 文件
- **客户端 (HTML5 + WebRTC API)**：
  - 选择本地视频文件并生成可发送的 `MediaStream`
  - 建立 WebRTC P2P 连接
  - 使用 `getStats()` 定时采集网络指标并上报

## 目录结构

```
.
├── index.html           # 前端页面（选择本地视频文件）
├── main.js              # WebRTC 逻辑 + stats 采集
├── server.js            # 信令与 CSV 写入服务
└── real_video_csv/      # 采集到的 CSV 输出目录（运行时自动创建）
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
   - 选择本地视频文件
   - 点击“加载本地视频并生成发送流”
   - 点击“建立连接并开始采集”

## P2P 通信流程简述

1. **信令交互**：通过 Socket.IO 交换 Offer/Answer
2. **ICE 交换**：通过 STUN 获取候选地址并互换
3. **媒体传输**：本地视频 `captureStream()` 生成 `MediaStream`，通过 `RTCPeerConnection.addTrack()` 发送
4. **对端渲染**：对端在 `ontrack` 中将流挂载到 `remoteVideo`

## CSV 数据输出

- 输出目录：`real_video_csv/`
- 文件命名：`webrtc_network_traces_<scenario>_<sessionId>.csv`
- 表头字段：
```
timestamp,clientId,rtt_ms,jitter,loss_rate,recv_bps,send_bps,estimated_bw_bps
```

## 关键代码位置

- 本地视频生成流：`main.js` 中 `startBtn` 点击逻辑
- 发送流接入：`main.js` 中 `setupRTC()` 的 `addTrack`
- 统计采集：`main.js` 中 `startDataCollection()`
- CSV 写入：`server.js` 中 `trace_data` 事件处理

---

如需增加：音频轨、循环播放、固定码率/分辨率等能力，可继续扩展。
