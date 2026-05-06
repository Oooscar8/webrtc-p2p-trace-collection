(function () {
    const STATE_ORDER = ['send_bps', 'recv_bps', 'rtt_ms', 'loss_rate', 'jitter_ms'];
    const STATE_SCALE_BY_COL = {
        send_bps: 1e-6,
        recv_bps: 1e-6,
        rtt_ms: 1e-2,
        loss_rate: 1.0,
        jitter_ms: 1e-2,
    };
    const NORMALIZATION_CLIP_RANGE = [-10.0, 10.0];
    const SEND_BPS_IDX = 0;
    const RECV_BPS_IDX = 1;
    const RTT_MS_IDX = 2;
    const LOSS_RATE_IDX = 3;
    const JITTER_MS_IDX = 4;
    const ONLINE_CLEAN_LOOKBACK = 3;
    const ONLINE_CLEAN_MAX_SHORT_GAP = 2;
    const ACTIVITY_BPS_THRESHOLD = 30_000;
    const MIN_POSITIVE_RTT_MS = 1.0;
    const MIN_POSITIVE_JITTER_MS = 0.1;

    function buildStateScales(stateCols) {
        return stateCols.map((col) => {
            if (!(col in STATE_SCALE_BY_COL)) {
                throw new Error(`Missing fixed scale for column: ${col}`);
            }
            return STATE_SCALE_BY_COL[col];
        });
    }

    function clamp(value, minValue, maxValue) {
        return Math.min(Math.max(value, minValue), maxValue);
    }

    function median(values) {
        if (!values.length) return null;
        const sorted = values.slice().sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        if (sorted.length % 2 === 1) return sorted[mid];
        return (sorted[mid - 1] + sorted[mid]) / 2;
    }

    function sanitizeObservation(obs) {
        return [
            Math.max(Number(obs[SEND_BPS_IDX]) || 0, 0),
            Math.max(Number(obs[RECV_BPS_IDX]) || 0, 0),
            Math.max(Number(obs[RTT_MS_IDX]) || 0, 0),
            clamp(Number(obs[LOSS_RATE_IDX]) || 0, 0, 1),
            Math.max(Number(obs[JITTER_MS_IDX]) || 0, 0),
        ];
    }

    function recentStates(history, lookback) {
        if (!history.length) return [];
        return history.slice(-lookback);
    }

    function hasRecentActivity(history) {
        return recentStates(history, ONLINE_CLEAN_LOOKBACK).some((state) => {
            return state[SEND_BPS_IDX] > ACTIVITY_BPS_THRESHOLD
                || state[RECV_BPS_IDX] > ACTIVITY_BPS_THRESHOLD
                || state[LOSS_RATE_IDX] > 0
                || state[RTT_MS_IDX] >= MIN_POSITIVE_RTT_MS
                || state[JITTER_MS_IDX] >= MIN_POSITIVE_JITTER_MS;
        });
    }

    function recentFeatureMedian(history, featureIdx, minimumPositive) {
        const values = recentStates(history, ONLINE_CLEAN_LOOKBACK)
            .map((state) => Number(state[featureIdx]) || 0)
            .filter((value) => value > minimumPositive);
        return median(values);
    }

    function replaceSuspiciousZero(featureIdx, currentValue, history, lastValidObs, recentActive) {
        if (currentValue > 0) {
            return { value: currentValue, suspicious: false };
        }

        let minimumPositive = 0;
        if (featureIdx === RTT_MS_IDX) minimumPositive = MIN_POSITIVE_RTT_MS;
        if (featureIdx === JITTER_MS_IDX) minimumPositive = MIN_POSITIVE_JITTER_MS;

        const recentMedian = recentFeatureMedian(history, featureIdx, minimumPositive);
        const lastValid = lastValidObs && Number(lastValidObs[featureIdx]) > minimumPositive
            ? Number(lastValidObs[featureIdx])
            : null;
        const replacement = recentMedian != null ? recentMedian : lastValid;
        const hadRecentValid = replacement != null;

        if (featureIdx === SEND_BPS_IDX || featureIdx === RECV_BPS_IDX) {
            const transportActive = recentActive
                || (lastValidObs && (
                    Number(lastValidObs[RTT_MS_IDX]) >= MIN_POSITIVE_RTT_MS
                    || Number(lastValidObs[JITTER_MS_IDX]) >= MIN_POSITIVE_JITTER_MS
                    || Number(lastValidObs[LOSS_RATE_IDX]) > 0
                ));
            if (transportActive && hadRecentValid) {
                return { value: replacement, suspicious: true };
            }
            return { value: currentValue, suspicious: false };
        }

        if ((featureIdx === RTT_MS_IDX || featureIdx === JITTER_MS_IDX) && recentActive && hadRecentValid) {
            return { value: replacement, suspicious: true };
        }

        return { value: currentValue, suspicious: false };
    }

    function buildWindowedState(history, norm) {
        if (!history.length) {
            throw new Error('Empty policy history');
        }
        const padded = history.slice(-norm.window_size);
        while (padded.length < norm.window_size) {
            padded.unshift(padded[0].slice());
        }
        const window = [];
        for (let featureIdx = 0; featureIdx < norm.state_cols.length; featureIdx += 1) {
            window.push(padded.map((state) => state[featureIdx]));
        }
        return window;
    }

    function flattenFeatureMajor(window) {
        const flat = [];
        for (const row of window) {
            for (const value of row) {
                flat.push(value);
            }
        }
        return flat;
    }

    function standardizeWindow(window, norm) {
        const featureMean = norm.feature_mean || [];
        const featureStd = norm.feature_std || [];
        if (!featureMean.length || !featureStd.length) {
            return window;
        }

        if (featureMean.length === window.length) {
            return window.map((row, featureIdx) => row.map((value) => {
                return (value - featureMean[featureIdx]) / (featureStd[featureIdx] + 1e-6);
            }));
        }

        const flat = flattenFeatureMajor(window);
        if (featureMean.length !== flat.length) {
            throw new Error(`Unsupported normalization shape: mean=${featureMean.length} flat=${flat.length}`);
        }

        const standardized = flat.map((value, idx) => {
            return (value - featureMean[idx]) / (featureStd[idx] + 1e-6);
        });

        const restored = [];
        let offset = 0;
        for (let rowIdx = 0; rowIdx < window.length; rowIdx += 1) {
            const row = standardized.slice(offset, offset + window[rowIdx].length);
            restored.push(row);
            offset += window[rowIdx].length;
        }
        return restored;
    }

    function transformWindow(window, norm) {
        let outWindow = window.map((row) => row.slice());
        if (norm.normalization_method === 'legacy_log1p_standardize') {
            const sendBps = outWindow[0].map((v) => Math.log1p(Math.max(v, 0.0) / 1e5));
            const recvBps = outWindow[1].map((v) => Math.log1p(Math.max(v, 0.0) / 1e5));
            const rtt = outWindow[2].map((v) => Math.log1p(Math.max(v, 0.0)));
            const loss = outWindow[3].map((v) => Math.min(Math.max(v, 0.0), 1.0));
            const jitter = outWindow[4].map((v) => Math.log1p(Math.max(v, 0.0)));
            outWindow = [sendBps, recvBps, rtt, loss, jitter];
            outWindow = standardizeWindow(outWindow, norm);
        } else {
            outWindow = outWindow.map((row, featureIdx) => {
                return row.map((value) => {
                    const scaled = value * norm.state_scales[featureIdx];
                    return Math.min(Math.max(scaled, norm.clip_min), norm.clip_max);
                });
            });
            outWindow = standardizeWindow(outWindow, norm);
        }

        const flat = Float32Array.from(flattenFeatureMajor(outWindow));
        if (flat.length !== norm.state_dim) {
            throw new Error(`state_dim mismatch: expected ${norm.state_dim}, got ${flat.length}`);
        }
        return flat;
    }

    function inferRawActionBps(session, inputName, outputName, x, norm) {
        const shape = norm.actor_type === 'gaussian' ? [1, 1, norm.state_dim] : [1, norm.state_dim];
        const obs = new ort.Tensor('float32', x, shape);
        return session.run({ [inputName]: obs }).then((outputs) => {
            const output = outputs[outputName];
            if (!output || !output.data || !output.data.length) {
                throw new Error('empty ONNX output');
            }
            const rawActionMbps = Number(output.data[0]);
            return rawActionMbps * norm.action_scale_bps;
        });
    }

    function loadNormFromJson(cfg) {
        const stateCols = Array.isArray(cfg.state_cols) && cfg.state_cols.length ? cfg.state_cols.map(String) : STATE_ORDER.slice();
        const windowSize = Number(cfg.window_size || 1);
        const stateDim = Number(cfg.state_dim || (stateCols.length * windowSize));
        const normalizationMethod = String(
            cfg.normalization_method || ((cfg.a_min !== undefined || cfg.a_max !== undefined)
                ? 'legacy_log1p_standardize'
                : 'schaferct_fixed_scale_clip')
        );
        const actionScaleBps = Number(cfg.action_scale_bps || 1_000_000.0);
        const actionMinBps = Number(cfg.a_min !== undefined ? cfg.a_min : (1e-5 * actionScaleBps));
        const actionMaxBps = Number(cfg.a_max !== undefined ? cfg.a_max : (cfg.max_action_bps !== undefined ? cfg.max_action_bps : Number.POSITIVE_INFINITY));
        return {
            state_cols: stateCols,
            window_size: windowSize,
            state_dim: stateDim,
            state_layout: String(cfg.state_layout || 'feature_major'),
            normalization_method: normalizationMethod,
            feature_mean: Array.isArray(cfg.feature_mean) ? cfg.feature_mean.map(Number) : [],
            feature_std: Array.isArray(cfg.feature_std) ? cfg.feature_std.map(Number) : [],
            state_scales: Array.isArray(cfg.state_scales) ? cfg.state_scales.map(Number) : buildStateScales(stateCols),
            clip_min: Number(cfg.clip_min !== undefined ? cfg.clip_min : NORMALIZATION_CLIP_RANGE[0]),
            clip_max: Number(cfg.clip_max !== undefined ? cfg.clip_max : NORMALIZATION_CLIP_RANGE[1]),
            observations_normalized: Boolean(cfg.observations_normalized),
            action_scale_bps: actionScaleBps,
            action_min_bps: actionMinBps,
            action_max_bps: actionMaxBps,
            actor_type: String(cfg.actor_type || 'deterministic'),
        };
    }

    class BrowserPolicyRuntime {
        constructor(session, norm) {
            this.session = session;
            this.norm = norm;
            this.inputName = session.inputNames[0];
            this.outputName = session.outputNames[0];
            this.history = [];
            this.lastActionBps = null;
            this.lastValidObs = null;
            this.consecutiveAnomalyCount = 0;
        }

        static async create(options) {
            if (typeof ort === 'undefined') {
                throw new Error('onnxruntime-web not loaded');
            }

            ort.env.wasm.numThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1));
            const normResp = await fetch(options.normUrl, { cache: 'no-store' });
            if (!normResp.ok) {
                throw new Error(`failed to load norm: ${normResp.status}`);
            }
            const normJson = await normResp.json();
            const norm = loadNormFromJson(normJson);
            if (norm.state_layout !== 'feature_major') {
                throw new Error(`Unsupported state layout: ${norm.state_layout}`);
            }

            const session = await ort.InferenceSession.create(options.modelUrl, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
            });
            return new BrowserPolicyRuntime(session, norm);
        }

        reset() {
            this.history = [];
            this.lastActionBps = null;
            this.lastValidObs = null;
            this.consecutiveAnomalyCount = 0;
        }

        cleanObservation(obs) {
            const sanitized = sanitizeObservation(obs);
            const recentActive = hasRecentActivity(this.history)
                || sanitized[SEND_BPS_IDX] > ACTIVITY_BPS_THRESHOLD
                || sanitized[RECV_BPS_IDX] > ACTIVITY_BPS_THRESHOLD
                || sanitized[LOSS_RATE_IDX] > 0
                || sanitized[RTT_MS_IDX] >= MIN_POSITIVE_RTT_MS
                || sanitized[JITTER_MS_IDX] >= MIN_POSITIVE_JITTER_MS;

            const cleaned = sanitized.slice();
            let suspiciousCount = 0;
            [SEND_BPS_IDX, RECV_BPS_IDX, RTT_MS_IDX, JITTER_MS_IDX].forEach((featureIdx) => {
                const result = replaceSuspiciousZero(
                    featureIdx,
                    cleaned[featureIdx],
                    this.history,
                    this.lastValidObs,
                    recentActive
                );
                cleaned[featureIdx] = result.value;
                if (result.suspicious) suspiciousCount += 1;
            });

            if (suspiciousCount > 0) {
                this.consecutiveAnomalyCount += 1;
            } else {
                this.consecutiveAnomalyCount = 0;
            }

            if (this.consecutiveAnomalyCount > ONLINE_CLEAN_MAX_SHORT_GAP) {
                this.history = [];
                this.consecutiveAnomalyCount = 0;
            }

            this.lastValidObs = cleaned.slice();
            return cleaned;
        }

        async predict(request) {
            const state = request.state || {};
            const obs = [
                Number(state.send_bps || 0),
                Number(state.recv_bps || 0),
                Number(state.rtt_ms || 0),
                Number(state.loss_rate || 0),
                Number(state.jitter_ms || 0),
            ];

            const isValid = obs.every(Number.isFinite) && obs[3] >= 0 && obs[3] <= 1;
            if (!isValid) {
                const fallback = Number(request.fallbackActionBps || request.prevActionBps || this.lastActionBps || 200000);
                this.lastActionBps = fallback;
                return {
                    action_bps: fallback,
                    raw_action_bps: fallback,
                    clipped: false,
                    smoothed: false,
                    fallback_used: true,
                };
            }

            const cleanedObs = this.cleanObservation(obs);
            this.history.push(cleanedObs);
            if (this.history.length > this.norm.window_size) {
                this.history = this.history.slice(-this.norm.window_size);
            }

            const window = buildWindowedState(this.history, this.norm);
            const x = transformWindow(window, this.norm);
            const raw = await inferRawActionBps(this.session, this.inputName, this.outputName, x, this.norm);

            let action = raw;
            let clipped = false;
            if (action < this.norm.action_min_bps) {
                action = this.norm.action_min_bps;
                clipped = true;
            } else if (action > this.norm.action_max_bps) {
                action = this.norm.action_max_bps;
                clipped = true;
            }

            const prev = request.prevActionBps != null ? Number(request.prevActionBps) : this.lastActionBps;
            let smoothed = false;
            if (prev != null && Number.isFinite(prev)) {
                const etaUp = 0.1;
                const etaDown = 0.3;
                if (action >= prev) {
                    action = (1 - etaUp) * prev + etaUp * action;
                    action = Math.min(action, prev * 1.25);
                } else {
                    action = (1 - etaDown) * prev + etaDown * action;
                    action = Math.max(action, prev * 0.65);
                }
                smoothed = true;
            }

            const fallback = Number(request.fallbackActionBps);
            if (Number.isFinite(fallback) && fallback > 0 && (action > fallback * 2.0 || action < fallback * 0.3)) {
                this.lastActionBps = fallback;
                return {
                    action_bps: fallback,
                    raw_action_bps: raw,
                    clipped,
                    smoothed,
                    fallback_used: true,
                };
            }

            this.lastActionBps = action;
            return {
                action_bps: action,
                raw_action_bps: raw,
                clipped,
                smoothed,
                fallback_used: false,
            };
        }
    }

    window.BrowserPolicyRuntime = BrowserPolicyRuntime;
}());
