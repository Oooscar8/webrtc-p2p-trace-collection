const express = require('express');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http);
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

app.use(express.json({ limit: '1mb' }));
app.use(express.static(__dirname));

// 初始化一个 Set 记录已创建的 CSV 文件
const activeCsvFiles = new Set();
const header = 'timestamp,clientId,rtt_ms,jitter,loss_rate,recv_bps,send_bps,estimated_bw_bps\n';
const outputDir = path.join(__dirname, '..', 'real_video_csv');

if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

function safeScenario(raw) {
    const s = String(raw || 'baseline')
        .toLowerCase()
        .replace(/[^a-z0-9_-]/g, '_')
        .slice(0, 64);
    return s || 'baseline';
}

function safeTraceStartTs(raw) {
    const n = Number(raw);
    if (!Number.isFinite(n)) return null;
    const v = Math.trunc(n);
    if (v <= 0) return null;
    return v;
}

let lastAutoNetworkUpdate = null;
const autoScriptPath = process.env.AUTO_SCRIPT_PATH || path.join(__dirname, '..', 'auto_collect_mac.sh');
let autoScriptProcess = null;

function forwardLines(stream, prefix, logFn) {
    if (!stream) return;
    stream.setEncoding('utf8');
    let buffer = '';
    stream.on('data', (chunk) => {
        buffer += chunk;
        const parts = buffer.split('\n');
        buffer = parts.pop() || '';
        for (const line of parts) {
            const t = line.trimEnd();
            if (t.length) logFn(`${prefix}${t}`);
        }
    });
    stream.on('end', () => {
        const t = buffer.trimEnd();
        if (t.length) logFn(`${prefix}${t}`);
        buffer = '';
    });
}

function startAutoScript() {
    if (autoScriptProcess && !autoScriptProcess.killed) {
        return { ok: true, status: 'running' };
    }
    if (!fs.existsSync(autoScriptPath)) {
        return { ok: false, error: `script not found: ${autoScriptPath}` };
    }
    try {
        autoScriptProcess = spawn('/bin/bash', [autoScriptPath], { stdio: ['ignore', 'pipe', 'pipe'] });
    } catch (error) {
        autoScriptProcess = null;
        return { ok: false, error: String(error) };
    }
    forwardLines(autoScriptProcess.stdout, '[auto_collect] ', console.log);
    forwardLines(autoScriptProcess.stderr, '[auto_collect][err] ', console.error);
    autoScriptProcess.on('error', (err) => {
        console.error('自动脚本启动失败:', err);
        autoScriptProcess = null;
    });
    autoScriptProcess.on('exit', () => {
        autoScriptProcess = null;
    });
    return { ok: true, status: 'started' };
}

app.post('/auto/start', (req, res) => {
    const result = startAutoScript();
    if (!result.ok) {
        res.status(500).json(result);
        return;
    }
    res.json(result);
});

app.post('/auto/network', (req, res) => {
    const scenario = safeScenario(req.body && req.body.scenario);
    const traceStartTs = safeTraceStartTs(req.body && req.body.traceStartTs);
    if (!traceStartTs) {
        res.status(400).json({ ok: false, error: 'invalid traceStartTs' });
        return;
    }
    lastAutoNetworkUpdate = { scenario, traceStartTs };
    io.emit('auto_network_update', lastAutoNetworkUpdate);
    res.json({ ok: true, scenario, traceStartTs });
});

io.on('connection', (socket) => {
    console.log('节点已连接:', socket.id);
    if (lastAutoNetworkUpdate) socket.emit('auto_network_update', lastAutoNetworkUpdate);

    // 1. 处理 WebRTC 信令及会话分配
    // 每当有一个新的 caller 进来时，生成一个唯一的信令组 (这里用一个简单的全局变量管理双端的会话ID)
    socket.on('message', (message) => {
        // 如果是发送 offer (说明是发起方发起了一次全新的通话呼叫)
        if (message.type === 'offer') {
            const currentSessionId = Math.floor(Math.random() * 1000000).toString();
            // 先告诉全体（包括自己和对方），这个新连接要用这个统一的 SessionID
            io.emit('set_session_id', currentSessionId);
            if (message.autoStart) startAutoScript();
        }
        socket.broadcast.emit('message', message);
    });

    // 2. 接收前端发来的网络轨迹数据，并追加到不同场景的 CSV 文件
    socket.on('trace_data', (data) => {
        let scenario = safeScenario(data && data.scenario);
        let traceStartTs = safeTraceStartTs(data && data.traceStartTs);
        const autoRunning = Boolean(autoScriptProcess && !autoScriptProcess.killed);
        if (autoRunning && lastAutoNetworkUpdate) {
            scenario = lastAutoNetworkUpdate.scenario;
            traceStartTs = lastAutoNetworkUpdate.traceStartTs;
        }
        const sessionSuffix = String((data && data.sessionId) || 'default').replace(/[^a-z0-9_-]/gi, '_').slice(0, 64);
        const csvFile = traceStartTs
            ? path.join(outputDir, `webrtc_network_traces_${scenario}_${traceStartTs}.csv`)
            : path.join(outputDir, `webrtc_network_traces_${scenario}_${sessionSuffix}.csv`);

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
