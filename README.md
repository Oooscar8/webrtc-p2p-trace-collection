# WebRTC 采集部署验证平台

本项目用于采集 WebRTC 视频通话过程中的底层网络状态数据（RTT、抖动、丢包率、带宽预估等），并进行离线强化学习（RL）模型训练与 A/B 测试验证。

---

## 完整端到端流程

```
┌─────────────────┐
│ 1. 数据集采集    │ → real_video_csv/*.csv
│ (GCC 轨迹)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. 生成训练集    │ → rl_dataset/transitions.npz
│ (四元组)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. 训练 IQL     │ → models/iql/actor.pt
│ 强化学习模型     │   models/iql/norm.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. 部署推理服务  │ → 监听端口 8000
│ (HTTP/WebSocket) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. A/B 测试验证  │ → ab_test_csv/*.csv
│ (GCC vs RL)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. QoE 报告     │ → output/qoe_ab_summary.csv
│ 对比分析        │
└─────────────────┘
```

---

## 项目架构

```
.
├── trace_collection/       # 数据集采集（GCC 轨迹）
│   ├── index.html
│   ├── main.js
│   └── server.js
├── ab_test/                # A/B Test（RL vs GCC）
│   ├── ab_test.html
│   ├── ab_test.js
│   └── server_abtest.js
├── rl/                     # RL 训练与推理服务
│   ├── train_iql.py       # IQL 离线训练脚本
│   ├── serve_policy.py    # 推理服务（HTTP/WebSocket）
│   └── requirements.txt   # Python 依赖
├── tools/                  # 工具脚本
│   ├── build_rl_dataset.py  # 从 CSV 生成 RL 训练集
│   └── qoe_report.py        # 生成 QoE 对比报告
├── models/                 # 模型文件（训练后生成）
│   └── iql/
│       ├── actor.pt        # Actor 网络（用于推理）
│       ├── critics.pt      # Critic 网络（仅训练用）
│       └── norm.json       # 归一化参数
├── real_video_csv/         # 数据集采集 CSV 输出
├── ab_test_csv/            # A/B Test CSV 输出
├── rl_dataset/             # RL 训练集输出
├── auto_collect_mac.sh     # 自动切换网络环境（Mac）
└── auto_fluctuate_mac.sh   # 波动网络脚本（可选）
```

---

## 详细步骤说明

### 步骤 1：数据集采集（GCC 轨迹）

采集浏览器默认 GCC 拥塞控制算法的网络轨迹数据。

#### 1.1 安装依赖
```bash
npm install
```

#### 1.2 启动采集服务（端口 3000）
```bash
# 如果需要自动切换网络，使用 sudo
sudo npm run start:collect
```

#### 1.3 浏览器打开页面
两台设备均打开：
```
http://<服务器IP>:3000
```

#### 1.4 开始采集
在每端浏览器执行：
- 选择模式（手动/自动）
- 选择本地视频文件
- 点击"加载本地视频并生成发送流"
- **仅在一侧**点击"建立连接并开始采集"

#### 输出
CSV 文件保存到：`real_video_csv/webrtc_network_traces_<scenario>_<traceStartTs>.csv`

---

### 步骤 2：生成 RL 训练集（四元组）

将采集到的 CSV 数据转换为强化学习训练用的四元组格式 `(s, a, r, s')`。

#### 2.1 生成训练集
```bash
python3 tools/build_rl_dataset.py \
  --input real_video_csv \
  --output rl_dataset \
  --window-size 10
```

#### 2.2 四元组定义
- **状态 s_t**：最近 `window_size` 行的聚合（默认 mean），包含 `[send_bps, recv_bps, rtt_ms, loss_rate, jitter_ms]`
- **动作 a_t**：当前行的 `estimated_bw_bps`
- **奖励 r_t**：QoE 形式：`log(1+recv_bps/1e6) - rtt_ms/1000 - loss_rate - 0.1*|a_t-a_{t-1}|/1e6`
- **下一状态 s_{t+1}**：窗口向后滑动 1 行

#### 输出
- `rl_dataset/transitions.npz`：训练用的四元组数据
- `rl_dataset/transitions.csv`：带元信息的 CSV
- `rl_dataset/manifest.csv`：统计信息

---

### 步骤 3：训练 IQL 强化学习模型

使用离线强化学习算法 IQL（Implicit Q-Learning）训练策略网络。

#### 3.1 安装 Python 依赖
```bash
python3 -m pip install -r rl/requirements.txt
```

#### 3.2 开始训练
```bash
python3 rl/train_iql.py \
  --dataset rl_dataset/transitions.npz \
  --outdir models/iql
```

#### 3.3 训练参数（可选）
- `--steps`：训练步数（默认 300000）
- `--batch`：Batch size（默认 256）
- `--lr`：学习率（默认 3e-4）
- `--expectile`：IQL 的 expectile 参数（默认 0.7）
- `--beta`：优势加权系数（默认 3.0）

#### 输出
- `models/iql/actor.pt`：**Actor 网络**（用于推理部署）
- `models/iql/critics.pt`：Critic 网络（仅用于训练，部署不需要）
- `models/iql/norm.json`：归一化参数

---

### 步骤 4：部署推理服务（HTTP / WebSocket）

启动推理服务，供 A/B Test 前端调用。

#### 4.1 启动服务（端口 8000）
```bash
python3 rl/serve_policy.py \
  --model models/iql/actor.pt \
  --norm models/iql/norm.json \
  --port 8000
```

#### 4.2 接口说明

**HTTP 接口**：
```
POST http://<host>:8000/predict
Content-Type: application/json

{
  "state": {
    "send_bps": 1000000,
    "recv_bps": 1000000,
    "rtt_ms": 50,
    "loss_rate": 0.01,
    "jitter_ms": 5
  },
  "prev_action_bps": 2000000,
  "fallback_action_bps": 2000000
}

Response:
{
  "action_bps": 2500000,
  "raw_action_bps": 2600000,
  "clipped": false,
  "smoothed": true,
  "fallback_used": false
}
```

**WebSocket 接口**：
```
ws://<host>:8000/ws
```

---

### 步骤 5：A/B 测试验证（GCC vs RL）

对比 GCC 和 RL 限速两种拥塞控制算法的 QoE 差异。

#### 5.1 确保推理服务已启动
确认端口 8000 上的 `serve_policy.py` 正在运行。

#### 5.2 启动 A/B Test 服务（端口 3001）
```bash
# 如果需要自动切换网络，使用 sudo
sudo npm run start:ab
```

#### 5.3 浏览器打开页面
两台设备均打开：
```
http://<服务器IP>:3001/ab_test.html
```

#### 5.4 配置 AB 分组
在页面中选择分组模式：
- **对照组：GCC**：仅使用浏览器 GCC
- **实验组：RL + 限速**：使用 RL 模型决策
- **随机 (50/50)**：每次连接随机分配（推荐用于 A/B 测试）

#### 5.5 开始采集
- 选择模式（手动/自动）
- 选择本地视频文件
- 点击"加载本地视频并生成发送流"
- **仅在一侧**点击"建立连接并开始采集"

#### 输出
CSV 文件保存到：`ab_test_csv/webrtc_abtest_traces_<scenario>_<traceStartTs>.csv`

---

### 步骤 6：生成 QoE 对比报告

分析 A/B Test 数据，生成 GCC vs RL 的 QoE 对比报告。

#### 6.1 生成报告
```bash
python3 tools/qoe_report.py \
  --input ab_test_csv \
  --outdir output
```

#### 6.2 QoE 指标
报告包含以下指标：
- `recv_bps_mean`：平均接收带宽（吞吐）
- `rtt_ms_p95`：95 分位 RTT（时延）
- `loss_rate_mean`：平均丢包率
- `jitter_ms_p95`：95 分位抖动
- `cap_delta_bps_mean`：码率变化平滑性
- `qoe_score_mean`：综合 QoE 分数

#### 6.3 输出文件
- `output/qoe_segments.csv`：每条 trace 按 client 统计的 QoE 指标
- `output/qoe_ab_summary.csv`：按 `(scenario, ab_group)` 聚合的 A/B 对比表

---

## 平台两种功能（完全隔离）

本仓库同时支持两条完全隔离的链路：

| 功能 | 输出目录 | 用途 |
|------|----------|------|
| **数据集采集** | `real_video_csv/` | 用于离线 RL 数据集构建与训练 |
| **A/B Test** | `ab_test_csv/` | 用于部署模型在线验证与 QoE 对比 |

> 建议：两条链路用不同端口启动不同 Node 服务，避免误写同目录/误读同 CSV。

---

## 自动网络脚本与权限

自动脚本需要 root 权限执行 `dnctl/pfctl`。推荐使用以下方式之一：
- 使用 `sudo npm run start:collect` 或 `sudo npm run start:ab` 启动服务端
- 或在系统中为 `dnctl/pfctl` 配置免密 sudo

可配置的环境变量：
- `AUTO_SCRIPT_PATH`：自动脚本路径（默认 `./auto_collect_mac.sh`）
- `SERVER_URL`：脚本通知服务端的地址（数据集采集默认 `http://localhost:3000`，A/B Test 默认 `http://localhost:3001`）
- `INTERVAL_SECONDS`：切换间隔秒数（默认 `300`）

---

## 关键代码位置

### 数据集采集（GCC 轨迹）
- 本地视频生成流：`trace_collection/main.js` 中 `startBtn` 点击逻辑
- 发送流接入：`trace_collection/main.js` 中 `setupRTC()` 的 `addTrack`
- 统计采集：`trace_collection/main.js` 中 `startDataCollection()`
- CSV 写入：`trace_collection/server.js` 中 `trace_data` 事件处理

### A/B Test（RL vs GCC）
- AB 分组逻辑：`ab_test/ab_test.js`
- RL 策略请求：`ab_test/ab_test.js`
- CSV 写入：`ab_test/server_abtest.js` 中 `trace_data` 事件处理

### RL 训练与推理
- IQL 训练：`rl/train_iql.py`
- 推理服务：`rl/serve_policy.py`
- 数据集生成：`tools/build_rl_dataset.py`
- QoE 报告：`tools/qoe_report.py`

---

## Actor vs Critic 模型说明

| 模型 | 用途 | 保存文件 | 部署时是否需要 |
|------|------|----------|----------------|
| **Actor** | 实际用于决策的策略网络，输入状态输出动作 | `actor.pt` | ✅ **需要** |
| **Critic** | 训练时用于评估动作价值（Q1/Q2/V） | `critics.pt` | ❌ **不需要** |

---

如需增加：音频轨、循环播放、固定码率/分辨率等能力，可继续扩展。
