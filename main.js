// const socket = io();
// 将 socket 连接指向 ngrok 暴露出来的 HTTPS 地址，避免混合内容导致浏览器阻止
const socket = io('https://actinometrical-snaringly-zola.ngrok-free.dev', { transports: ['websocket'] });
let localStream, peerConnection;

// 独立会话 ID，用于区分同一场景下的多次采集 (通过房间会话级别统一下发，此处不再随机生成)
let sessionId = 'default';

// 用于计算速率的全局变量
let lastBytesReceived = 0, lastBytesSent = 0, lastTime = 0;
let collectorInterval;

document.getElementById('startBtn').onclick = async () => {
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
        alert('请先选择一个本地视频文件');
        return;
    }

    const localVideo = document.getElementById('localVideo');

    // 使用本地视频文件生成 MediaStream
    const url = URL.createObjectURL(file);
    localVideo.src = url;
    localVideo.muted = true;

    try {
        await localVideo.play();
    } catch (err) {
        console.error('本地视频播放失败:', err);
        alert('本地视频播放失败，请更换文件或检查浏览器权限设置');
        URL.revokeObjectURL(url);
        return;
    }

    // 从视频元素捕获流（较新的浏览器支持）
    if (typeof localVideo.captureStream === 'function') {
        localStream = localVideo.captureStream();
    } else if (typeof localVideo.mozCaptureStream === 'function') {
        localStream = localVideo.mozCaptureStream();
    } else {
        alert('当前浏览器不支持从视频元素捕获流，请使用新版 Chrome/Edge/Firefox');
        URL.revokeObjectURL(url);
        return;
    }

    document.getElementById('callBtn').disabled = false;
};

document.getElementById('callBtn').onclick = async () => {
    setupRTC();
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    socket.emit('message', { type: 'offer', offer: offer });
};

// 信令处理逻辑 (简化版)
// 追加一个专用的事件，用来同步全局/双端的 Session ID
socket.on('set_session_id', (id) => {
    sessionId = id;
    console.log(`[同步] 当前录制会话 ID 已统一为: ${sessionId}`);
});

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

        // 获取用户选择的当前网络场景
        const scenario = document.getElementById('scenarioSelect').value;
        trace.scenario = scenario;
        trace.sessionId = sessionId.toString();

        // 将数据发给 Node.js 保存
        socket.emit('trace_data', trace);
        
    }, 500); // 500ms 采集一次
}