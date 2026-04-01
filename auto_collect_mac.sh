#!/usr/bin/env bash
set -euo pipefail

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "❌ 错误: 请使用 sudo 运行此脚本！"
  echo "👉 sudo ./auto_collect_mac.sh"
  exit 1
fi

SERVER_URL="${SERVER_URL:-http://localhost:3000}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"

cleanup() {
  echo -e "\n\n=== 正在恢复网络状态并退出 ==="
  dnctl -q flush || true
  pfctl -f /etc/pf.conf 2>/dev/null || true
  pfctl -d 2>/dev/null || true
  echo "✅ 退出成功，Mac 网络已恢复正常。"
  exit 0
}

trap cleanup SIGINT SIGTERM SIGQUIT

now_ms() {
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import time; print(int(time.time()*1000))'
    return 0
  fi
  echo "$(( $(date +%s) * 1000 ))"
}

notify_frontend() {
  local scenario="$1"
  local traceStartTs="$2"
  local payload
  payload="$(printf '{"scenario":"%s","traceStartTs":%s}' "$scenario" "$traceStartTs")"
  if ! curl -sS --connect-timeout 2 --max-time 5 --retry 2 --retry-all-errors -X POST "${SERVER_URL}/auto/network" -H 'Content-Type: application/json' -d "${payload}" >/dev/null; then
    echo "⚠️  [warn] notify_frontend failed: scenario=${scenario} traceStartTs=${traceStartTs}" >&2
  fi
}

apply_profile() {
  local scenario="$1"
  case "${scenario}" in
    baseline)
      # Downlink Bandwidth: 0 Kbps (无限制), Downlink Delay: 0 ms, Downlink Loss: 0%
      # Uplink Bandwidth: 0 Kbps (无限制), Uplink Delay: 0 ms, Uplink Loss: 0%
      dnctl pipe 1 config bw 1000Mbit/s delay 0ms plr 0.00
      ;;
    3g)
      # Downlink Bandwidth: 780 Kbps, Downlink Delay: 100 ms, Downlink Loss: 0%
      # Uplink Bandwidth: 330 Kbps, Uplink Delay: 100 ms, Uplink Loss: 0%
      # 取上下行带宽的平均值作为统一带宽
      dnctl pipe 1 config bw 555Kbit/s delay 100ms plr 0.00
      ;;
    low_bw)
      # Downlink Bandwidth: 500 Kbps, Downlink Delay: 0 ms, Downlink Loss: 0%
      # Uplink Bandwidth: 500 Kbps, Uplink Delay: 0 ms, Uplink Loss: 0%
      dnctl pipe 1 config bw 500Kbit/s delay 0ms plr 0.00
      ;;
    high_loss)
      # Downlink Bandwidth: 10 Mbps, Downlink Delay: 0 ms, Downlink Loss: 5%
      # Uplink Bandwidth: 10 Mbps, Uplink Delay: 0 ms, Uplink Loss: 5%
      dnctl pipe 1 config bw 10Mbit/s delay 0ms plr 0.05
      ;;
    high_delay)
      # Downlink Bandwidth: 5 Mbps, Downlink Delay: 150 ms, Downlink Loss: 0%
      # Uplink Bandwidth: 5 Mbps, Uplink Delay: 150 ms, Uplink Loss: 0%
      dnctl pipe 1 config bw 5Mbit/s delay 150ms plr 0.00
      ;;
    fluctuating)
      # 波动网络：按照 auto_fluctuate_mac.sh 的逻辑
      # 正常WiFi波动->蜂窝网络基站切换->逐渐拥塞与逐渐恢复->偶发的纯丢包环境->极端的短暂随机高丢包->正常WiFi波动
      # 初始状态: 20Mbps, 延迟 10ms, 丢包 0%
      dnctl pipe 1 config bw 20Mbit/s delay 10ms plr 0.00
      sleep 15
      # 状态切换: 有人下载大文件 (3Mbps, 延迟 50ms, 丢包 1%)
      dnctl pipe 1 config bw 3Mbit/s delay 50ms plr 0.01
      sleep 8
      # 状态切换: 移动到极弱WiFi角落 (800Kbps, 延迟 120ms, 丢包 3%)
      dnctl pipe 1 config bw 800Kbit/s delay 120ms plr 0.03
      sleep 15
      # 状态切换: 5G 良好覆盖 (50Mbps, 延迟 30ms, 丢包 0%)
      dnctl pipe 1 config bw 50Mbit/s delay 30ms plr 0.00
      sleep 15
      # 状态切换: 进电梯/隧道瞬断 (100Kbps, 延迟 300ms, 丢包 10%)
      dnctl pipe 1 config bw 100Kbit/s delay 300ms plr 0.10
      sleep 4
      # 状态切换: 刚出隧道降级为 3G 状态 (1Mbps, 延迟 100ms, 丢包 0%)
      dnctl pipe 1 config bw 1Mbit/s delay 100ms plr 0.00
      sleep 15
      # 状态切换: 正常 (10Mbps, 10ms, 0%)
      dnctl pipe 1 config bw 10Mbit/s delay 10ms plr 0.00
      sleep 10
      # 状态切换: 轻度拥塞 (5Mbps, 20ms, 0%)
      dnctl pipe 1 config bw 5Mbit/s delay 20ms plr 0.00
      sleep 5
      # 状态切换: 中度拥塞 (2Mbps, 40ms, 1%)
      dnctl pipe 1 config bw 2Mbit/s delay 40ms plr 0.01
      sleep 5
      # 状态切换: 重度极弱网 (500Kbps, 80ms, 2%)
      dnctl pipe 1 config bw 500Kbit/s delay 80ms plr 0.02
      sleep 5
      # 状态切换: 拥塞缓解，逐步恢复 (2Mbps, 20ms, 0%)
      dnctl pipe 1 config bw 2Mbit/s delay 20ms plr 0.00
      sleep 8
      # 状态切换: 高速低延迟 (50Mbps, 10ms, 0%)
      dnctl pipe 1 config bw 50Mbit/s delay 10ms plr 0.00
      sleep 10
      # 状态切换: 极端的短暂随机高丢包 (50Mbps, 10ms, 8% 丢包)
      dnctl pipe 1 config bw 50Mbit/s delay 10ms plr 0.08
      sleep 7
      # 状态切换: 正常WiFi波动 (20Mbps, 延迟 10ms, 丢包 0%)
      dnctl pipe 1 config bw 20Mbit/s delay 10ms plr 0.00
      ;;
    lte)
      # Downlink Bandwidth: 50 Mbps, Downlink Delay: 50 ms, Downlink Loss: 0%
      # Uplink Bandwidth: 10 Mbps, Uplink Delay: 66 ms, Uplink Loss: 0%
      # 取上下行带宽的平均值作为统一带宽，取上下行延迟的平均值作为统一延迟
      dnctl pipe 1 config bw 30Mbit/s delay 58ms plr 0.00
      ;;
    dsl)
      # Downlink Bandwidth: 2 Mbps, Downlink Delay: 5 ms, Downlink Loss: 0%
      # Uplink Bandwidth: 256 Kbps, Uplink Delay: 5 ms, Uplink Loss: 0%
      # 取上下行带宽的平均值作为统一带宽
      dnctl pipe 1 config bw 1128Kbit/s delay 5ms plr 0.00
      ;;
    very_bad)
      # Downlink Bandwidth: 1 Mbps, Downlink Delay: 500 ms, Downlink Loss: 10%
      # Uplink Bandwidth: 1 Mbps, Uplink Delay: 500 ms, Uplink Loss: 10%
      dnctl pipe 1 config bw 1Mbit/s delay 500ms plr 0.10
      ;;
    *)
      dnctl pipe 1 config bw 1000Mbit/s delay 0ms plr 0.00
      ;;
  esac
}

echo "=== 自动化采集模式：每 ${INTERVAL_SECONDS}s 切换一次网络环境（9 种循环）==="
echo "按下 Ctrl+C 随时停止并恢复网络"
echo "SERVER_URL=${SERVER_URL}"
echo ""

pfctl -e 2>/dev/null || true
dnctl -q flush
echo "dummynet out all pipe 1
dummynet in all pipe 1" | pfctl -f - 2>/dev/null
dnctl pipe 1 config bw 1000Mbit/s delay 0ms plr 0.00

scenarios=(baseline fluctuating very_bad lte dsl 3g low_bw high_loss high_delay)

while true; do
  for scenario in "${scenarios[@]}"; do
    start_s="$(date +%s)"
    traceStartTs="$(now_ms)"
    echo "[$(date +%T)] 切换网络环境: ${scenario}  (traceStartTs=${traceStartTs})"
    dnctl pipe 1 config bw 1000Mbit/s delay 0ms plr 0.00
    notify_frontend "${scenario}" "${traceStartTs}"
    apply_profile "${scenario}"
    elapsed_s="$(( $(date +%s) - start_s ))"
    remaining_s="$(( INTERVAL_SECONDS - elapsed_s ))"
    if [ "${remaining_s}" -gt 0 ]; then
      sleep "${remaining_s}"
    fi
  done
done
