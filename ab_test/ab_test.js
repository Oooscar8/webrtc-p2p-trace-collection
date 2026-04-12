const socket = io({ transports: ['polling', 'websocket'] });
let localStream, peerConnection;
let localVideoEndHandlerInstalled = false;

// 独立会话 ID，用于区分同一场景下的多次采集 (通过房间会话级别统一下发，此处不再随机生成)
let sessionId = 'default';

// A/B group (session-scoped)
let abGroup = 'unknown';
let originalAbMode = 'random';

// 用于计算速率的全局变量
let lastBytesReceived = 0, lastBytesSent = 0, lastTime = 0;
let collectorInterval;

const scenarioSelect = document.getElementById('scenarioSelect');
const modeRadios = document.querySelectorAll('input[name="collectMode"]');

const enablePolicyEl = document.getElementById('enablePolicy');
const policyServerUrlEl = document.getElementById('policyServerUrl');
const appliedMaxBitrateEl = document.getElementById('appliedMaxBitrate');
const abModeEl = document.getElementById('abMode');
const abGroupDisplayEl = document.getElementById('abGroupDisplay');

const POLICY_REQUEST_MIN_INTERVAL_MS = 1000;
const POLICY_REQUEST_TIMEOUT_MS = 400;

let videoSender = null;
let appliedMaxBitrateBps = null;
let lastPolicyActionBps = null;
let lastPolicyRequestAt = 0;
let policyInFlight = false;

let collectMode = 'manual';
let activeScenario = scenarioSelect.value;
let activeTraceStartTs = null;
let lastAutoNetworkUpdate = null;

if (policyServerUrlEl && !policyServerUrlEl.value) {
    const host = window.location.hostname || '127.0.0.1';
    const proto = window.location.protocol || 'http:';
    policyServerUrlEl.value = `${proto}//${host}:8000`;
}

function normalizeAbGroup(raw) {
    const v = String(raw || '').toLowerCase();
    if (v === 'rl' || v === 'gcc') return v;
    return 'gcc';
}

function setAbGroup(group, disableSelect = true) {
    abGroup = normalizeAbGroup(group);
    if (abGroupDisplayEl) abGroupDisplayEl.textContent = abGroup;
    if (enablePolicyEl) enablePolicyEl.checked = abGroup === 'rl';
    if (abModeEl) {
        abModeEl.value = abGroup;
        if (disableSelect) {
            abModeEl.disabled = true;
        }
    }

    if (abGroup !== 'rl') {
        applyMaxBitrate(videoSender, null);
    }
}

function shouldUsePolicy() {
    return Boolean(enablePolicyEl && enablePolicyEl.checked && abGroup === 'rl');
}

function getPolicyServerBaseUrl() {
    if (!policyServerUrlEl) return '';
    const raw = String(policyServerUrlEl.value || '').trim();
    return raw.endsWith('/') ? raw.slice(0, -1) : raw;
}

function resetRateCounters() {
    lastBytesReceived = 0;
    lastBytesSent = 0;
    lastTime = 0;
}

function beginNewTraceSegment({ scenario, traceStartTs }) {
    activeScenario = scenario || scenarioSelect.value || 'baseline';
    activeTraceStartTs = traceStartTs || Date.now();
    resetRateCounters();
}

function setCollectMode(mode) {
    collectMode = mode === 'auto' ? 'auto' : 'manual';
    scenarioSelect.disabled = collectMode === 'auto';
    if (collectMode === 'auto' && lastAutoNetworkUpdate) {
        if (lastAutoNetworkUpdate.scenario) scenarioSelect.value = lastAutoNetworkUpdate.scenario;
        beginNewTraceSegment({ scenario: scenarioSelect.value, traceStartTs: lastAutoNetworkUpdate.traceStartTs });
    }
}

function setAppliedMaxBitrateText(bps) {
    if (!appliedMaxBitrateEl) return;
    if (!bps || !Number.isFinite(bps)) {
        appliedMaxBitrateEl.textContent = '-';
        return;
    }
    appliedMaxBitrateEl.textContent = String(Math.round(bps / 1000));
}

function clampMaxBitrateBps(bps) {
    const v = Math.round(Number(bps));
    if (!Number.isFinite(v)) return null;
    return Math.min(Math.max(v, 30_000), 20_000_000);
}

async function applyMaxBitrate(sender, maxBitrateBps) {
    if (!sender) return;
    const params = sender.getParameters();
    if (!params.encodings || params.encodings.length === 0) params.encodings = [{}];

    if (maxBitrateBps == null) {
        if (params.encodings[0] && 'maxBitrate' in params.encodings[0]) {
            delete params.encodings[0].maxBitrate;
        }
        try {
            await sender.setParameters(params);
        } catch (err) {
            console.warn('clear maxBitrate failed:', err);
        }
        appliedMaxBitrateBps = null;
        lastPolicyActionBps = null;
        setAppliedMaxBitrateText(null);
        return;
    }

    const clamped = clampMaxBitrateBps(maxBitrateBps);
    if (clamped == null) return;

    params.encodings[0].maxBitrate = clamped;
    try {
        await sender.setParameters(params);
        appliedMaxBitrateBps = clamped;
        setAppliedMaxBitrateText(clamped);
    } catch (err) {
        console.warn('set maxBitrate failed:', err);
    }
}

async function requestPolicyAction({ state, prevActionBps, fallbackActionBps }) {
    const baseUrl = getPolicyServerBaseUrl();
    if (!baseUrl) throw new Error('empty policy server url');

    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), POLICY_REQUEST_TIMEOUT_MS);
    try {
        const resp = await fetch(`${baseUrl}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                state,
                prev_action_bps: prevActionBps ?? null,
                fallback_action_bps: fallbackActionBps ?? null,
            }),
            signal: controller.signal,
        });
        if (!resp.ok) throw new Error(`policy http ${resp.status}`);
        const out = await resp.json();
        const action = Number(out && out.action_bps);
        if (!Number.isFinite(action)) throw new Error('invalid policy response');
        return action;
    } finally {
        clearTimeout(t);
    }
}

modeRadios.forEach(radio => {
    radio.addEventListener('change', () => {
        if (!radio.checked) return;
        setCollectMode(radio.value);
        if (collectMode === 'manual') beginNewTraceSegment({ scenario: scenarioSelect.value, traceStartTs: Date.now() });
    });
});

scenarioSelect.addEventListener('change', () => {
    activeScenario = scenarioSelect.value;
    if (collectMode === 'manual') beginNewTraceSegment({ scenario: activeScenario, traceStartTs: Date.now() });
});

socket.on('auto_network_update', (payload) => {
    lastAutoNetworkUpdate = payload || null;
    if (collectMode !== 'auto') return;
    const scenario = payload && payload.scenario;
    const traceStartTs = payload && payload.traceStartTs;
    if (scenario) scenarioSelect.value = scenario;
    beginNewTraceSegment({ scenario: scenarioSelect.value, traceStartTs });
});

// 信令处理逻辑 (简化版)
socket.on('set_session_id', (id) => {
    sessionId = id;
    console.log(`[同步] 当前录制会话 ID 已统一为: ${sessionId}`);
});

socket.on('set_ab_group', (group) => {
    setAbGroup(group, false);
    console.log(`[同步] 当前 AB 分组已统一为: ${abGroup}`);
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
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });

    if (localStream) {
        localStream.getTracks().forEach(track => {
            const sender = peerConnection.addTrack(track, localStream);
            if (track.kind === 'video') videoSender = sender;
        });
    }

    peerConnection.ontrack = e => {
        document.getElementById('remoteVideo').srcObject = e.streams[0];
        if (!collectorInterval) startDataCollection();
    };

    peerConnection.onicecandidate = e => {
        if (e.candidate) socket.emit('message', { type: 'candidate', candidate: e.candidate });
    };
}

document.getElementById('startBtn').onclick = async () => {
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
        alert('请先选择一个本地视频文件');
        return;
    }

    const localVideo = document.getElementById('localVideo');
    const url = URL.createObjectURL(file);
    localVideo.src = url;
    localVideo.muted = true;
    localVideo.loop = true;
    if (!localVideoEndHandlerInstalled) {
        localVideo.addEventListener('ended', () => {
            localVideo.currentTime = 0;
            localVideo.play().catch(() => {});
        });
        localVideoEndHandlerInstalled = true;
    }

    try {
        await localVideo.play();
    } catch (err) {
        console.error('本地视频播放失败:', err);
        alert('本地视频播放失败，请更换文件或检查浏览器权限设置');
        URL.revokeObjectURL(url);
        return;
    }

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

    const abMode = abModeEl ? String(abModeEl.value || '').toLowerCase() : 'random';
    originalAbMode = abMode;
    const chosenAbGroup = abMode === 'random' ? (Math.random() < 0.5 ? 'gcc' : 'rl') : normalizeAbGroup(abMode);
    setAbGroup(chosenAbGroup);

    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    socket.emit('message', {
        type: 'offer',
        offer: offer,
        autoStart: collectMode === 'auto',
        abGroup: chosenAbGroup,
        abMode: abMode,
    });
};

function startDataCollection() {
    if (!activeTraceStartTs) beginNewTraceSegment({ scenario: scenarioSelect.value, traceStartTs: Date.now() });
    collectorInterval = setInterval(async () => {
        if (!peerConnection || peerConnection.connectionState !== 'connected') return;

        const stats = await peerConnection.getStats(null);
        let trace = {
            timestamp: Date.now(),
            rtt: 0,
            jitter: 0,
            lossRate: 0,
            recvBitrate: 0,
            sendBitrate: 0,
            gccEstimatedBw: 0,
            policyMaxBitrateBps: 0,
            estimatedBw: 0,
            abGroup: abGroup,
        };
        let packetsLost = 0, packetsReceived = 0;

        stats.forEach(report => {
            if (report.type === 'candidate-pair' && report.state === 'succeeded') {
                trace.rtt = report.currentRoundTripTime ? report.currentRoundTripTime * 1000 : 0;
                trace.gccEstimatedBw = report.availableOutgoingBitrate || 0;
                trace.estimatedBw = trace.gccEstimatedBw;
            }
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
            if (report.type === 'outbound-rtp' && report.kind === 'video') {
                if (lastTime > 0) {
                    const bytesDiff = report.bytesSent - lastBytesSent;
                    const timeDiff = (report.timestamp - lastTime) / 1000;
                    trace.sendBitrate = Math.round((bytesDiff * 8) / timeDiff);
                }
                lastBytesSent = report.bytesSent || 0;
            }
        });

        const totalPackets = packetsReceived + packetsLost;
        const lossRateValue = totalPackets > 0 ? packetsLost / totalPackets : 0;
        trace.lossRate = lossRateValue.toFixed(4);

        lastTime = performance.timeOrigin + performance.now();

        const gccEstimatedBw = trace.gccEstimatedBw;

        if (shouldUsePolicy() && videoSender) {
            const now = Date.now();

            if (appliedMaxBitrateBps != null) {
                trace.estimatedBw = appliedMaxBitrateBps;
            }

            const shouldRequest =
                !policyInFlight && now - lastPolicyRequestAt >= POLICY_REQUEST_MIN_INTERVAL_MS;

            if (shouldRequest) {
                policyInFlight = true;
                lastPolicyRequestAt = now;

                const state = {
                    send_bps: Number(trace.sendBitrate) || 0,
                    recv_bps: Number(trace.recvBitrate) || 0,
                    rtt_ms: Number(trace.rtt) || 0,
                    loss_rate: Number(lossRateValue) || 0,
                    jitter_ms: Number(trace.jitter) || 0,
                };

                requestPolicyAction({
                    state,
                    prevActionBps: lastPolicyActionBps,
                    fallbackActionBps: gccEstimatedBw,
                })
                    .then((actionBps) => {
                        lastPolicyActionBps = actionBps;
                        return applyMaxBitrate(videoSender, actionBps);
                    })
                    .catch((err) => {
                        console.warn('policy request failed:', err);
                    })
                    .finally(() => {
                        policyInFlight = false;
                    });
            }
        } else {
            if (appliedMaxBitrateBps != null) {
                appliedMaxBitrateBps = null;
                setAppliedMaxBitrateText(null);
            }
        }

        trace.policyMaxBitrateBps = appliedMaxBitrateBps || 0;
        trace.abGroup = abGroup;

        trace.scenario = activeScenario;
        trace.traceStartTs = activeTraceStartTs;
        trace.sessionId = sessionId.toString();

        socket.emit('ab_trace_data', trace);

    }, 500);
}