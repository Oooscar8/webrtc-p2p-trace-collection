// const socket = io();
// 将 socket 连接指向 ngrok 暴露出来的 HTTPS 地址，避免混合内容导致浏览器阻止
const socket = io('https://actinometrical-snaringly-zola.ngrok-free.dev', { transports: ['websocket'] });
let localStream, peerConnection;

// 用于计算速率的全局变量
let lastBytesReceived = 0, lastBytesSent = 0, lastTime = 0;
let collectorInterval;

document.getElementById('startBtn').onclick = async () => {
    // 使用 HTML5 Canvas 伪造一个视频流代替真实摄像头
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    
    // 定时绘制颜色变化的画面，确保 H.264/VP8 编码器有实际内容可发送
    let color = 0;
    setInterval(() => {
        ctx.fillStyle = `hsl(${color % 360}, 100%, 50%)`;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        color += 5;
    }, 33); // 约 30 FPS 的刷新率

    // 从 canvas 捕获视频流，指定每秒帧数
    localStream = canvas.captureStream(30);

    document.getElementById('localVideo').srcObject = localStream;
    document.getElementById('callBtn').disabled = false;
};

document.getElementById('callBtn').onclick = async () => {
    setupRTC();
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    socket.emit('message', { type: 'offer', offer: offer });
};

// 信令处理逻辑 (简化版)
socket.on('message', async (data) => {
    if (data.type === 'offer') {
        setupRTC();
        await peerConnection.setRemoteDescription(data.offer);
        const answer = await peerConnection.createAnswer();
        await peerConnection.setLocalDescription(answer);
        socket.emit('message', { type: 'answer', answer: answer });
    } else if (data.type === 'answer') {
        await peerConnection.setRemoteDescription(data.answer);
    } else if (data.type === 'candidate' && peerConnection) {
        await peerConnection.addIceCandidate(data.candidate);
    }
});

function setupRTC() {
    peerConnection = new RTCPeerConnection({
        iceServers: [
            { urls: 'stun:stun.l.google.com:19302' }
            // 在不同 NAT / 跨公网时配置可用的 TURN
            // { urls: 'turn:<turn-host>:3478', username: '<u>', credential: '<p>' }
        ]
    });
    
    localStream.getTracks().forEach(track => peerConnection.addTrack(track, localStream));
    
    peerConnection.ontrack = e => {
        document.getElementById('remoteVideo').srcObject = e.streams[0];
        // 对方流接入后，开始采集数据
        if (!collectorInterval) startDataCollection();
    };

    peerConnection.onicecandidate = e => {
        if (e.candidate) socket.emit('message', { type: 'candidate', candidate: e.candidate });
    };
}

// 核心：定时采集网络状态并发送给服务器
function startDataCollection() {
    // 设置收集步长，例如 500ms (这对 RL state 来说是一个合理的决策间隔)
    collectorInterval = setInterval(async () => {
        if (!peerConnection || peerConnection.connectionState !== 'connected') return;

        const stats = await peerConnection.getStats(null);
        let trace = { timestamp: Date.now(), rtt: 0, jitter: 0, lossRate: 0, recvBitrate: 0, sendBitrate: 0, estimatedBw: 0 };
        let packetsLost = 0, packetsReceived = 0;

        stats.forEach(report => {
            // 1. RTT 和 GCC 预估可用带宽
            if (report.type === 'candidate-pair' && report.state === 'succeeded') {
                trace.rtt = report.currentRoundTripTime ? report.currentRoundTripTime * 1000 : 0;
                trace.estimatedBw = report.availableOutgoingBitrate || 0;
            }
            // 2. 接收端数据 (丢包、抖动、接收速率)
            if (report.type === 'inbound-rtp' && report.kind === 'video') {
                trace.jitter = report.jitter ? report.jitter * 1000 : 0;
                packetsLost = report.packetsLost || 0;
                packetsReceived = report.packetsReceived || 0;
                
                if (lastTime > 0) {
                    const bytesDiff = report.bytesReceived - lastBytesReceived;
                    const timeDiff = (report.timestamp - lastTime) / 1000;
                    trace.recvBitrate = Math.round((bytesDiff * 8) / timeDiff);
                }
                lastBytesReceived = report.bytesReceived || 0;
            }
            // 3. 发送端数据 (发送速率)
            if (report.type === 'outbound-rtp' && report.kind === 'video') {
                if (lastTime > 0) {
                    const bytesDiff = report.bytesSent - lastBytesSent;
                    const timeDiff = (report.timestamp - lastTime) / 1000;
                    trace.sendBitrate = Math.round((bytesDiff * 8) / timeDiff);
                }
                lastBytesSent = report.bytesSent || 0;
            }
        });

        // 计算丢包率
        const totalPackets = packetsReceived + packetsLost;
        if (totalPackets > 0) trace.lossRate = (packetsLost / totalPackets).toFixed(4);
        
        lastTime = performance.timeOrigin + performance.now();

        // 将数据发给 Node.js 保存
        socket.emit('trace_data', trace);
        
    }, 500); // 500ms 采集一次
}