# -*- coding: utf-8 -*-
import pigpio as pio
import time
import cv2
import numpy as np

# ===================== 硬件配置（和你小车完全一致） =====================
MOTOR_PIN = 13
STEER_PIN = 12

# 连接pigpio
pi = pio.pi()
if not pi.connected:
    print("❌ 请先运行: sudo pigpiod")
    exit()

# PWM设置
pi.set_PWM_frequency(MOTOR_PIN, 200)
pi.set_PWM_range(MOTOR_PIN, 40000)
pi.set_PWM_frequency(STEER_PIN, 50)
pi.set_PWM_range(STEER_PIN, 20000)

# 速度参数
SPEED_STOP = 10000
SPEED_FORWARD = 11500  # 自动直行速度

# 转向参数（全自动只需要直行回正）
STEER_CENTER = 1500

# ===================== 小车核心动作（无人控制专用） =====================
def car_stop():
    """小车停止"""
    pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_STOP)
    pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)

def car_forward():
    """小车直行"""
    pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
    pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_FORWARD)

# ===================== 颜色识别阈值 =====================
# 蓝色挡板
lower_blue = np.array([90, 60, 60])
upper_blue = np.array([130, 255, 255])

# 白色斑马线
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

# ===================== 摄像头初始化 =====================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 状态标记
zebra_paused = False  # 斑马线是否在暂停

# ===================== 【纯无人】主循环 =====================
print("="*50)
print("🤖 纯无人自动驾驶已启动")
print("1. 蓝色挡板 → 自动停车 | 移开 → 自动行驶")
print("2. 白色斑马线 → 停3秒 → 自动继续走")
print("按 Q 键退出程序")
print("="*50)

car_forward()  # 开机自动直行

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 图像预处理
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # 1. 检测蓝色挡板（最高优先级）
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_area = cv2.countNonZero(blue_mask)

        # 2. 检测白色斑马线
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_area = cv2.countNonZero(white_mask)

        # ============== 无人自动逻辑 ==============
        # 🔴 规则1：检测到蓝色 → 强制停车
        if blue_area > 2000:
            car_stop()
            print("🔵 检测到蓝色挡板 → 已停车")

        # 🟢 规则2：无蓝色 → 执行斑马线/正常行驶
        else:
            # 规则A：不在斑马线暂停 → 正常判断
            if not zebra_paused:
                # 检测到斑马线 → 停3秒
                if white_area > 1500:
                    car_stop()
                    print("⚪ 检测到斑马线 → 停车3秒")
                    zebra_paused = True
                    time.sleep(3)
                    car_forward()
                    print("⏰ 3秒结束 → 继续直行")
                    zebra_paused = False
                
                # 无斑马线 → 持续直行
                else:
                    car_forward()

            # 规则B：正在斑马线暂停 → 不执行任何动作
            else:
                pass

        # 显示画面
        cv2.imshow("无人自动驾驶画面", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 安全停止
    car_stop()
    cap.release()
    cv2.destroyAllWindows()
    pi.stop()
    print("\n✅ 程序退出，小车已安全停止")
