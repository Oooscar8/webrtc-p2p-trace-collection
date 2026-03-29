const express = require('express');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http);
const fs = require('fs');
const path = require('path');

app.use(express.static(__dirname));

// 初始化一个 Set 记录已创建的 CSV 文件
const activeCsvFiles = new Set();
const header = 'timestamp,clientId,rtt_ms,jitter,loss_rate,recv_bps,send_bps,estimated_bw_bps\n';
const outputDir = path.join(__dirname, 'real_video_csv');

if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

io.on('connection', (socket) => {
    console.log('节点已连接:', socket.id);

    // 1. 处理 WebRTC 信令及会话分配
    // 每当有一个新的 caller 进来时，生成一个唯一的信令组 (这里用一个简单的全局变量管理双端的会话ID)
    socket.on('message', (message) => {
        // 如果是发送 offer (说明是发起方发起了一次全新的通话呼叫)
        if (message.type === 'offer') {
            const currentSessionId = Math.floor(Math.random() * 1000000).toString();
            // 先告诉全体（包括自己和对方），这个新连接要用这个统一的 SessionID
            io.emit('set_session_id', currentSessionId);
        }
        socket.broadcast.emit('message', message);
    });

    // 2. 接收前端发来的网络轨迹数据，并追加到不同场景的 CSV 文件
    socket.on('trace_data', (data) => {
        const scenario = data.scenario || 'baseline';
        const sessionSuffix = data.sessionId || 'default';
        const csvFile = path.join(outputDir, `webrtc_network_traces_${scenario}_${sessionSuffix}.csv`);

        // 如果文件之前在这个运行会话中没被记录过，检查并创建表头
        if (!activeCsvFiles.has(csvFile)) {
            if (!fs.existsSync(csvFile)) {
                fs.writeFileSync(csvFile, header);
            }
            activeCsvFiles.add(csvFile);
        }

        const row = `${data.timestamp},${socket.id},${data.rtt},${data.jitter},${data.lossRate},${data.recvBitrate},${data.sendBitrate},${data.estimatedBw}\n`;
        fs.appendFile(csvFile, row, (err) => {
            if (err) console.error(`写入 ${csvFile} 失败`, err);
        });
    });
});

http.listen(3000, '0.0.0.0', () => {
    console.log('数据采集服务器运行在端口 3000');
});
