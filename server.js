const express = require('express');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http);
const fs = require('fs');

app.use(express.static(__dirname));

// 初始化 CSV 文件和表头
const csvFile = 'webrtc_network_traces.csv';
const header = 'timestamp,clientId,rtt_ms,jitter,loss_rate,recv_bps,send_bps,estimated_bw_bps\n';
if (!fs.existsSync(csvFile)) {
    fs.writeFileSync(csvFile, header);
}

io.on('connection', (socket) => {
    console.log('节点已连接:', socket.id);

    // 1. 处理 WebRTC 信令
    socket.on('message', (message) => {
        socket.broadcast.emit('message', message);
    });

    // 2. 接收前端发来的网络轨迹数据，并追加到 CSV 文件
    socket.on('trace_data', (data) => {
        const row = `${data.timestamp},${socket.id},${data.rtt},${data.jitter},${data.lossRate},${data.recvBitrate},${data.sendBitrate},${data.estimatedBw}\n`;
        fs.appendFile(csvFile, row, (err) => {
            if (err) console.error('写入 CSV 失败', err);
        });
    });
});

http.listen(3000, '0.0.0.0', () => {
    console.log('数据采集服务器运行在端口 3000');
});