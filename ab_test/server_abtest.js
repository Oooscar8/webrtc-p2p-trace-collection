const express = require('express');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http);
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

app.use(express.json({ limit: '1mb' }));
app.use(express.static(__dirname));

// AB Test 输出目录（与 real_video_csv 完全隔离）
const activeCsvFiles = new Set();
const header = 'timestamp,clientId,ab_group,rtt_ms,jitter,loss_rate,recv_bps,send_bps,gcc_estimated_bw_bps,policy_max_bitrate_bps,estimated_bw_bps\n';
const outputDir = path.join(__dirname, '..', 'ab_test_csv');

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

function safeAbGroup(raw) {
    const v = String(raw || '').toLowerCase();
    if (v === 'rl' || v === 'gcc') return v;
    if (v === 'random') return Math.random() < 0.5 ? 'gcc' : 'rl';
    return 'gcc';
}

let lastAutoNetworkUpdate = null;
let lastAbGroup = null;
let lastAbMode = null;
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
    
    // 如果原始模式是 random，每次网络切换时重新随机分配 AB 分组
    if (lastAbMode === 'random') {
        lastAbGroup = safeAbGroup('random');
        io.emit('set_ab_group', lastAbGroup);
        console.log(`[abtest] 网络切换至 ${scenario}，重新随机分组为: ${lastAbGroup}`);
    }
    
    res.json({ ok: true, scenario, traceStartTs });
});

io.on('connection', (socket) => {
    console.log('[abtest] 节点已连接:', socket.id);
    if (lastAutoNetworkUpdate) socket.emit('auto_network_update', lastAutoNetworkUpdate);
    if (lastAbGroup) socket.emit('set_ab_group', lastAbGroup);

    // 1) 信令 + 会话级 AB 分组
    socket.on('message', (message) => {
        if (message.type === 'offer') {
            const currentSessionId = Math.floor(Math.random() * 1000000).toString();
            lastAbMode = String(message && message.abMode || '').toLowerCase();
            lastAbGroup = safeAbGroup(message && message.abGroup);
            io.emit('set_session_id', currentSessionId);
            io.emit('set_ab_group', lastAbGroup);
            if (message.autoStart) startAutoScript();
        }
        socket.broadcast.emit('message', message);
    });

    // 2) AB test trace 数据写入
    socket.on('ab_trace_data', (data) => {
        let scenario = safeScenario(data && data.scenario);
        let traceStartTs = safeTraceStartTs(data && data.traceStartTs);
        const autoRunning = Boolean(autoScriptProcess && !autoScriptProcess.killed);
        if (autoRunning && lastAutoNetworkUpdate) {
            scenario = lastAutoNetworkUpdate.scenario;
            traceStartTs = lastAutoNetworkUpdate.traceStartTs;
        }

        const sessionSuffix = String((data && data.sessionId) || 'default').replace(/[^a-z0-9_-]/gi, '_').slice(0, 64);
        const csvFile = traceStartTs
            ? path.join(outputDir, `webrtc_abtest_traces_${scenario}_${traceStartTs}.csv`)
            : path.join(outputDir, `webrtc_abtest_traces_${scenario}_${sessionSuffix}.csv`);

        if (!activeCsvFiles.has(csvFile)) {
            if (!fs.existsSync(csvFile)) {
                fs.writeFileSync(csvFile, header);
            }
            activeCsvFiles.add(csvFile);
        }

        const abGroup = safeAbGroup((data && data.abGroup) || lastAbGroup);

        const ts = Number(data && data.timestamp) || Date.now();
        const rtt = Number(data && data.rtt) || 0;
        const jitter = Number(data && data.jitter) || 0;
        const lossRate = Number(data && data.lossRate) || 0;
        const recvBitrate = Number(data && data.recvBitrate) || 0;
        const sendBitrate = Number(data && data.sendBitrate) || 0;
        const gccEstimatedBw = Number(data && data.gccEstimatedBw) || 0;
        const policyMaxBitrateBps = Number(data && data.policyMaxBitrateBps) || 0;
        const estimatedBw = Number(data && data.estimatedBw) || 0;

        const row = `${ts},${socket.id},${abGroup},${rtt},${jitter},${lossRate},${recvBitrate},${sendBitrate},${gccEstimatedBw},${policyMaxBitrateBps},${estimatedBw}\n`;
        fs.appendFile(csvFile, row, (err) => {
            if (err) console.error(`写入 ${csvFile} 失败`, err);
        });
    });
});

const port = Number(process.env.PORT) || 3001;
http.listen(port, '0.0.0.0', () => {
    console.log(`[abtest] 服务运行在端口 ${port}`);
});