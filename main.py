# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pigpio as pio
import time
import sys
import tty
import termios
import select

# ======================== 颜色阈值（保留原始全部定义） ========================
blue_lower = np.array([100, 43, 46])
blue_upper = np.array([124, 255, 255])
red_lower = np.array([170, 100, 100])
red_upper = np.array([179, 255, 255])
green_lower = np.array([35, 43, 46])
green_upper = np.array([77, 255, 255])

# ======================== 小车硬件配置 ========================
MOTOR_PIN = 13
STEER_PIN = 12

pi = pio.pi()
if not pi.connected:
    print("❌ 请先运行: sudo pigpiod")
    sys.exit(1)

pi.set_PWM_frequency(MOTOR_PIN, 200)
pi.set_PWM_range(MOTOR_PIN, 40000)
pi.set_PWM_frequency(STEER_PIN, 50)
pi.set_PWM_range(STEER_PIN, 20000)

SPEED_STOP = 10000
SPEED_FORWARD = 10900
SPEED_BACK = 9100

STEER_CENTER = 1500
STEER_LEFT = 2000
STEER_RIGHT = 1000

# 小车初始化
pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_STOP)
pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
time.sleep(0.2)

# ======================== 摄像头初始化 ========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 摄像头无法打开")
    pi.stop()
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 确认实际设置是否生效
actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"📷 摄像头实际分辨率: {actual_w}x{actual_h}, FPS: {actual_fps}")

# ======================== 终端设置（只设置一次） ========================
fd = sys.stdin.fileno()
old_termios = termios.tcgetattr(fd)
tty.setraw(fd)

def get_key_press():
    """非阻塞读取按键，不再反复切换终端模式"""
    rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
    return sys.stdin.read(1) if rlist else None

# ======================== 主循环 ========================
print("=" * 50)
print("🎮 切换模式：按下移动，松开即停")
print(" W=前进  S=后退  A=左转  D=右转  Q=退出")
print("=" * 50)

try:
    while True:
        # 1. 读取画面
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 帧读取失败")
            time.sleep(0.05)
            continue

        # 2. 颜色识别（轻量处理）
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, red_lower, red_upper)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius > 5:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                print(f"\r🎯 目标坐标: ({int(x):3d}, {int(y):3d})", end="")
            else:
                print("\r" + " " * 40, end="")
        else:
            print("\r" + " " * 40, end="")

        # 3. 显示画面
        cv2.imshow("Car Camera + Detect", frame)

        # 4. 键盘控制 —— 原始逻辑：有按键就执行，无按键就停车
        key = get_key_press()
        if key == "q":
            break
        elif key == "w":
            pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_FORWARD)
        elif key == "s":
            pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_BACK)
        elif key == "a":
            pi.set_PWM_dutycycle(STEER_PIN, STEER_LEFT)
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_FORWARD)
        elif key == "d":
            pi.set_PWM_dutycycle(STEER_PIN, STEER_RIGHT)
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_FORWARD)
        else:
            # 松开按键立即停车并回正（这就是你要的“切换模式”）
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_STOP)
            pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)

        # 5. 按 ESC 也可退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("\n⚠️ 用户中断")
finally:
    # 安全释放所有资源
    cap.release()
    cv2.destroyAllWindows()
    pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_STOP)
    pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
    pi.stop()
    termios.tcsetattr(fd, termios.TCSADRAIN, old_termios)
    print("\n✅ 程序安全退出")