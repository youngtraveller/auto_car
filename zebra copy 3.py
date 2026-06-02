# -*- coding: utf-8 -*-
"""
参考 car_race.py 的斑马线检测程序
"""
import cv2
import numpy as np
import pigpio as pio
import sys
import time
import os
import threading

# 硬件配置（与 car_race.py 完全一致）
MOTOR_PIN = 13
STEER_PIN = 12

pi = pio.pi()
if not pi.connected:
    print("ERROR: pigpio not connected!")
    print("Please run: sudo pigpiod")
    sys.exit(1)

# PWM 设置（与 car_race.py 完全一致）
pi.set_PWM_frequency(MOTOR_PIN, 200)
pi.set_PWM_range(MOTOR_PIN, 40000)
pi.set_PWM_frequency(STEER_PIN, 50)
pi.set_PWM_range(STEER_PIN, 20000)

# 核心 PWM 参数（与 car_race.py 完全一致）
SPEED_STOP    = 10000
SPEED_FORWARD = 10600
STEER_CENTER  = 1500

# 摄像头初始化（与 car_race.py 完全一致）
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 验证摄像头参数
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fcc_str = "".join([chr((actual_fcc >> i) & 0xFF) for i in (0, 8, 16, 24)])
fcc_str = ''.join(c if c.isprintable() else '?' for c in fcc_str)
print("Camera: {}x{} codec:{}".format(actual_w, actual_h, fcc_str))

FRAME_W = 160
FRAME_H = 120

# 斑马线检测参数
VERTICAL_LINE_TOLERANCE = 20
MIN_LINE_LENGTH = 20
MIN_LINES = 2
ROI_BOTTOM_RATIO = 0.5
ANTI_SHAKE_FRAMES = 2

PARK_DELAY = 10
VOICE_PATH = "/home/pi/test_img/voice_module/bin/zebra.wav"
COOLDOWN = 0.5

# 播放语音
def play_voice_async():
    print("Playing voice:", VOICE_PATH)
    def voice_thread():
        os.system("aplay {}".format(VOICE_PATH))
    threading.Thread(target=voice_thread, daemon=True).start()

# 全局状态
running = True
parking = False
park_start_time = 0
anti_shake_count = 0
last_trigger_time = 0
roi_y_start = int(FRAME_H * ROI_BOTTOM_RATIO)

print("\n=== Starting crosswalk detection ===")
print("Press Ctrl+C to stop")

# 先设置舵机中位和电机停止（与 car_race.py 一致的初始化顺序）
pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_STOP)
pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
time.sleep(0.2)

# 开始前进（直接使用 pi.set_PWM_dutycycle，与 car_race.py 一致）
pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_FORWARD)
print("Car started moving")

frame_count = 0

try:
    while running:
        frame_count += 1
        
        # 读取帧
        ok, frame = cap.read()
        if not ok:
            # 帧读取失败时继续保持前进
            pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_FORWARD)
            time.sleep(0.02)
            continue
        
        # 处理斑马线检测
        roi = frame[roi_y_start:FRAME_H, 0:FRAME_W]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, 40, 8)

        crosswalk_detected = False
        
        if lines is not None:
            vertical_count = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if abs(x1 - x2) < VERTICAL_LINE_TOLERANCE and length > MIN_LINE_LENGTH:
                    vertical_count += 1
            
            if vertical_count >= MIN_LINES:
                anti_shake_count += 1
                if anti_shake_count >= ANTI_SHAKE_FRAMES:
                    crosswalk_detected = True
                    print("Frame {}: CROSSWALK DETECTED!".format(frame_count))
            else:
                anti_shake_count = 0

        now = time.time()
        if not parking:
            # 正常行驶（直接设置 PWM，与 car_race.py 一致）
            pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_FORWARD)
            
            if crosswalk_detected and now - last_trigger_time > COOLDOWN:
                # 检测到斑马线，停车
                pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_STOP)
                pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
                parking = True
                park_start_time = now
                play_voice_async()
                print("=== Stopping for {} seconds ===".format(PARK_DELAY))
        else:
            # 停车状态
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_STOP)
            pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
            
            elapsed = now - park_start_time
            if elapsed >= PARK_DELAY:
                parking = False
                anti_shake_count = 0
                last_trigger_time = now
                print("=== Resume moving after {:.1f}s ===".format(elapsed))

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
finally:
    print("\n=== Cleaning up ===")
    pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_STOP)
    pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
    time.sleep(0.2)
    pi.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Done")
