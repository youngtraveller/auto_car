# -*- coding: utf-8 -*-
import pigpio as pio
import time
import sys
import tty
import termios
import select

# ===================== 硬件配置 =====================
MOTOR_PIN = 13
STEER_PIN = 12

# 连接pigpio
pi = pio.pi()
if not pi.connected:
    print("❌ 请先运行: sudo pigpiod")
    sys.exit()

# PWM设置
pi.set_PWM_frequency(MOTOR_PIN, 200)
pi.set_PWM_range(MOTOR_PIN, 40000)
pi.set_PWM_frequency(STEER_PIN, 50)
pi.set_PWM_range(STEER_PIN, 20000)

# ===================== 车速（正常使用） =====================
SPEED_STOP = 10000
SPEED_FORWARD = 10900  # 前进
SPEED_BACK = 9100     # 后退

# ===================== 【修复：舵机转向拉满】 =====================
# 标准舵机大角度，绝对能转向！
STEER_CENTER = 1500  # 居中
STEER_LEFT   = 2000  # 左转（角度加大）
STEER_RIGHT  = 1000  # 右转（角度加大）
# =================================================================

# 小车初始化
pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_STOP)
pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
time.sleep(0.2)

# 高性能键盘检测（长按动，松开停）
def get_key_press():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([fd], [], [], 0.01)
        return sys.stdin.read(1) if rlist else None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# 控制界面
print("="*50)
print("🎮 手动小车（长按动·松开停）✅ 前后左右正常")
print(" W=前进  S=后退  A=左转  D=右转  Q=退出")
print("="*50)

try:
    while True:
        key = get_key_press()

        # 无按键 → 立即停车+回正
        if key is None:
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_STOP)
            pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
            continue

        # 退出
        if key == 'q':
            break

        # 前进
        if key == 'w':
            pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_FORWARD)
            print("\r→ 前进中 ", end="")

        # 后退
        elif key == 's':
            pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_BACK)
            print("\r← 后退中 ", end="")

        # 左转（核心修复）
        elif key == 'a':
            pi.set_PWM_dutycycle(STEER_PIN, STEER_LEFT)
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_FORWARD)
            print("\r↖ 左转中 ", end="")

        # 右转（核心修复）
        elif key == 'd':
            pi.set_PWM_dutycycle(STEER_PIN, STEER_RIGHT)
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_FORWARD)
            print("\r↗ 右转中 ", end="")

        time.sleep(0.001)

finally:
    # 安全停止
    pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_STOP)
    pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
    pi.stop()
    print("\n✅ 小车已停止")