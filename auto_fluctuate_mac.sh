#!/bin/bash

# ==========================================
# 强制要求使用 sudo 运行，以防止权限导致的按键无响应
if [ "$EUID" -ne 0 ]; then
  echo "❌ 错误: 请使用 sudo 运行此脚本！"
  echo "👉 请在终端输入: sudo ./auto_fluctuate_mac.sh"
  exit 1
fi

# 定义清理并恢复网络的函数
cleanup() {
    echo -e "\n\n=== 正在恢复网络状态并退出 ==="
    sudo dnctl -q flush
    sudo pfctl -f /etc/pf.conf 2>/dev/null
    sudo pfctl -d 2>/dev/null
    echo "✅ 退出成功，Mac 网络已恢复正常。"
    exit 0
}

# 捕捉 Ctrl+C (SIGINT) 和强制终止信号 (SIGTERM), 一旦触发就调用 cleanup
trap cleanup SIGINT SIGTERM SIGQUIT

echo "=== 开始自动化 WebRTC 网络状态模拟 (挂机收集适用) ==="
echo "按下 Ctrl+C 随时停止并恢复原本网络"
echo "在此期间，您可以在浏览器页面点击刷新以生成新的 sessionId 从而分开记录数据集。"
echo ""

# 开启防火墙包过滤，并清空历史 dummynet 配置
sudo pfctl -e 2>/dev/null
sudo dnctl -q flush

# 将 Mac 上的所有网络流量映射到 dummynet 当中编号为 1 的规则管道 (pipe)
# 如果你发现电脑完全不能上网了，这就是流量整形生效的标志。
echo "dummynet out all pipe 1
dummynet in all pipe 1" | sudo pfctl -f - 2>/dev/null

# 随机等待 10-20 秒的函数
rand_sleep() {
    sleep_time=$((RANDOM % 11 + 10))
    echo "⏳ 保持当前状态 ${sleep_time} 秒..."
    sleep $sleep_time
}

# 循环挂机，自动化切换不同状态
while true; do
    echo -e "\n============================================="
    echo "[$(date +%T)] === 场景 1: 正常的家庭WiFi波动 (引入复合情况) ==="
    echo "  👉 初始状态: 20Mbps, 延迟 10ms, 丢包 0%"
    sudo dnctl pipe 1 config bw 20Mbit/s delay 10ms plr 0.00
    rand_sleep
    
    echo "  👉 状态切换: 有人下载大文件 (3Mbps, 延迟 50ms, 丢包 1%)"
    sudo dnctl pipe 1 config bw 3Mbit/s delay 50ms plr 0.01
    sleep 8
    
    echo "  👉 状态切换: 移动到极弱WiFi角落 (800Kbps, 延迟 120ms, 丢包 3%)"
    sudo dnctl pipe 1 config bw 800Kbit/s delay 120ms plr 0.03
    rand_sleep

    echo -e "\n============================================="
    echo "[$(date +%T)] === 场景 2: 蜂窝网络基站切换 (断崖与短暂恢复) ==="
    echo "  👉 初始状态: 5G 良好覆盖 (50Mbps, 延迟 30ms, 丢包 0%)"
    sudo dnctl pipe 1 config bw 50Mbit/s delay 30ms plr 0.00
    rand_sleep
    
    echo "  👉 状态切换: 进电梯/隧道瞬断 (100Kbps, 延迟 300ms, 丢包 10%)"
    sudo dnctl pipe 1 config bw 100Kbit/s delay 300ms plr 0.10
    sleep 4
    
    echo "  👉 状态切换: 刚出隧道降级为 3G 状态 (1Mbps, 延迟 100ms, 丢包 0%)"
    sudo dnctl pipe 1 config bw 1Mbit/s delay 100ms plr 0.00
    rand_sleep
    
    echo -e "\n============================================="
    echo "[$(date +%T)] === 场景 3: 逐渐拥塞与逐渐恢复 (Staircase) ==="
    echo "  👉 阶梯 1: 正常 (10Mbps, 10ms, 0%)"
    sudo dnctl pipe 1 config bw 10Mbit/s delay 10ms plr 0.00
    sleep 10
    
    echo "  👉 阶梯 2: 轻度拥塞 (5Mbps, 20ms, 0%)"
    sudo dnctl pipe 1 config bw 5Mbit/s delay 20ms plr 0.00
    sleep 5
    
    echo "  👉 阶梯 3: 中度拥塞 (2Mbps, 40ms, 1%)"
    sudo dnctl pipe 1 config bw 2Mbit/s delay 40ms plr 0.01
    sleep 5
    
    echo "  👉 阶梯 4: 重度极弱网 (500Kbps, 80ms, 2%)"
    sudo dnctl pipe 1 config bw 500Kbit/s delay 80ms plr 0.02
    sleep 5
    
    echo "  👉 阶梯 5: 拥塞缓解，逐步恢复 (2Mbps, 20ms, 0%)"
    sudo dnctl pipe 1 config bw 2Mbit/s delay 20ms plr 0.00
    sleep 8

    echo -e "\n============================================="
    echo "[$(date +%T)] === 场景 4: 偶发的纯丢包环境 (测试抗抖动性) ==="
    echo "  👉 初始状态: 高速低延迟 (50Mbps, 10ms, 0%)"
    sudo dnctl pipe 1 config bw 50Mbit/s delay 10ms plr 0.00
    sleep 10
    
    echo "  👉 状态切换: 极端的短暂随机高丢包 (50Mbps, 10ms, 8% 丢包)"
    sudo dnctl pipe 1 config bw 50Mbit/s delay 10ms plr 0.08
    sleep 7
done
