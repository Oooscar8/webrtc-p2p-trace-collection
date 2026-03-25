# WebRTC 强化学习网络轨迹采集终端 (WebRTC RL Collector)

本项目是一个用于采集 WebRTC 视频通话过程中底层网络状态数据（如 RTT、抖动、丢包率、带宽预估等）的终端系统。采集到的海量数据可用于训练强化学习（RL）模型，以优化网络拥塞控制或带宽探测算法。

## 项目架构

项目采用 **Client-Server（C/S）** 架构辅助建立 **Peer-to-Peer（P2P）** 通信：
- **服务端 (Node.js + Socket.IO)**: 负责提供静态页面托管、担任 WebRTC 交换 SDP 和 ICE 候选者的信令服务器（Signaling Server），以及负责接收前端回传的网络状态数据并持久化到本地 CSV 文件中。
- **客户端 (前端 HTML5 + WebRTC API)**: 运行在两端设备的浏览器中，负责生成高熵视频流、建立 WebRTC P2P 连接、以及通过 `getStats()` API 定时轮询 WebRTC 底层传输的统计数据并上报给服务端。

## 两端设备如何进行 P2P 通信？

两端设备通过浏览器内置的 WebRTC 技术进行点对点通信，建立连接的基本流程如下：

1. **信令交互 (Signaling)**: WebRTC 自身不包含信令控制协议，因此本项目通过 Socket.IO 实现信令服务器 (`server.js`)。
   - **发起方**点击“建立连接”后，创建一个 `RTCPeerConnection` 实例，生成一个包含本地媒体参数的 `Offer`（Session Description Protocol, SDP）。
   - 发起方将 `Offer` 发送给信令服务器，信令服务器将其转发（广播）给**接收方**。
   - 接收方收到 `Offer` 后，将其设置为远程描述，然后生成一个 `Answer` (SDP) 并通过信令服务器发回给发起方。
2. **网络穿透 (ICE Negotiation)**: 
   - 寻找自身在公网的 IP 和端口的过程。两端在创建 `RTCPeerConnection` 时配置了 STUN 服务器（如 `stun:stun.l.google.com:19302`）。
   - 两端浏览器会通过 STUN 服务器获取自己的公网地址（ICE Candidate），然后通过信令服务器将这些 Candidate 交换给对方。
   - 双方利用收集到的候选者信息打洞建立直连（如果处于复杂的 NAT 之后，可能还需要 TURN 服务器中继，本项目默认使用 STUN 尝试 P2P 直连）。
3. **媒体流传输**:
   - 双方成功连通后，开始通过建立的 P2P 通道传输视频流（`MediaStreamTrack`）。本项目为了有效触发带宽探测，在 `main.js` 中利用 Canvas 绘制不断变化的高熵随机噪点画面，模拟高码率需求的视频流。

## 获取与记录网络轨迹数据到 CSV

前端采集数据并由后端写入 CSV，流程如下：

1. **前端定时采集（`main.js`）**:
   - 当 P2P 连接成功且远端视频流开始播放时，前端启动一个定时器 (`setInterval`)，每隔 **500 毫秒**调用一次 `peerConnection.getStats(null)`。
   - 遍历 `getStats` 返回的报告，提取所需指标：
     - **RTT 与预估带宽**: 从类型为 `candidate-pair`（且 `state === 'succeeded'`）的报告中解析当回合 RTT (`currentRoundTripTime`) 和底层 GCC 算法预估的输出带宽 (`availableOutgoingBitrate`)。
     - **接收端数据**: 从 `inbound-rtp` 的视频类型报告中获取当前接收的字节数 (`bytesReceived`)、抖动 (`jitter`)、丢包数 (`packetsLost`) 等计算出接收速率（Receive Bitrate）和丢包率。
     - **发送端数据**: 从 `outbound-rtp` 报告中获取当前发送的字节数 (`bytesSent`)，计算发送速率（Send Bitrate）。
   - 将收集到的各项指标连同时间戳（`timestamp`）、当前模拟场景标识（`scenario`）和会话 ID（`sessionId`）打包成一个 JSON 对象，通过 `socket.emit('trace_data', trace)` 发送给服务器。

2. **后端写入 CSV（`server.js`）**:
   - 服务器监听 `trace_data` 事件。
   - 接收到前端发来的 JSON 数据后，根据传入的 `scenario` 和 `sessionId` 动态决定要写入的文件名（例如 `webrtc_network_traces_baseline_123456.csv`）。
   - 服务器维护一个 `activeCsvFiles` 集合，如果是首次遇到某个文件，则先写入 CSV 表头 (`timestamp,clientId,rtt_ms,jitter,loss_rate,recv_bps,send_bps,estimated_bw_bps`)。
   - 然后，后端把传来的数据按照表头字段拼接成一行逗号分隔的字符串 (`row`)。
   - 利用 Node.js 的 `fs.appendFile` 方法，将该行数据无阻塞地追加写入到对应的 CSV 文件中，从而完成网络轨迹的持久化沉淀，供后续 RL 模型训练与离线分析使用。