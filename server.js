const express = require('express');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http);
const fs = require('fs');

app.use(express.static(__dirname));

// 初始化一个 Set 记录已创建的 CSV 文件
const activeCsvFiles = new Set();
const header = 'timestamp,clientId,rtt_ms,jitter,loss_rate,recv_bps,send_bps,estimated_bw_bps\n';

io.on('connection', (socket) => {
    console.log('节点已连接:', socket.id);

    // 1. 处理 WebRTC 信令
    socket.on('message', (message) => {
        socket.broadcast.emit('message', message);
    });

    // 2. 接收前端发来的网络轨迹数据，并追加到不同场景的 CSV 文件
    socket.on('trace_data', (data) => {
        const scenario = data.scenario || 'baseline';
        const csvFile = `webrtc_network_traces_${scenario}.csv`;

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