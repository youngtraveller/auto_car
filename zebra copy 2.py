# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pigpio as pio
import sys
import time
import os
import threading

# Hardware Configuration (same as car_race.py)
MOTOR_PIN = 13
STEER_PIN = 12

pi = pio.pi()
if not pi.connected:
    print("ERROR: pigpio not connected!")
    print("Please run: sudo pigpiod")
    sys.exit(1)

# PWM settings (same as car_race.py)
pi.set_PWM_frequency(MOTOR_PIN, 200)
pi.set_PWM_range(MOTOR_PIN, 40000)
pi.set_PWM_frequency(STEER_PIN, 50)
pi.set_PWM_range(STEER_PIN, 20000)

# Speed settings (EXACT same as car_race.py)
SPEED_STOP    = 10000                     # 电机停止
SPEED_FORWARD = 10600                     # 前进速度（稳定不冲）
STEER_CENTER  = 1500                      # 舵机中位（直行）

# Camera initialization - EXACT same as car_race.py
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Verify settings
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fcc_str = "".join([chr((actual_fcc >> i) & 0xFF) for i in (0, 8, 16, 24)])
fcc_str = ''.join(c if c.isprintable() else '?' for c in fcc_str)
print("Camera: {}x{} codec:{}".format(actual_w, actual_h, fcc_str))

if not cap.isOpened():
    print("ERROR: Camera NOT opened!")
    pi.stop()
    sys.exit(1)
print("Camera opened successfully!")

FRAME_W = 160
FRAME_H = 120

# Crosswalk detection parameters
VERTICAL_LINE_TOLERANCE = 20
MIN_LINE_LENGTH = 20
MIN_LINES = 2
ROI_BOTTOM_RATIO = 0.5
ANTI_SHAKE_FRAMES = 2

PARK_DELAY = 10
VOICE_PATH = "/home/pi/test_img/voice_module/bin/zebra.wav"
COOLDOWN = 0.5

# Display settings
SHOW_WINDOW = True
DISPLAY_EVERY_N = 5

# Global state
running = True
parking = False
park_start_time = 0
anti_shake_count = 0
last_trigger_time = 0
roi_y_start = int(FRAME_H * ROI_BOTTOM_RATIO)
frame_count = 0

print("\n=== Starting crosswalk detection ===")
print("Press Ctrl+C to stop")

# ================ 关键修复：立即设置电机和舵机初始状态 ================
# 参考 car_race.py 的做法：在主循环前先设置初始状态
print("Setting initial motor and steer state...")
pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)  # 舵机回正
pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_FORWARD) # 立即启动前进
time.sleep(0.1)  # 给电机一点响应时间
print("Car started moving")

def play_voice_async():
    def voice_thread():
        os.system("aplay {}".format(VOICE_PATH))
    threading.Thread(target=voice_thread, daemon=True).start()

try:
    while running:
        frame_count += 1
        
        # Read frame
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        
        # Process ROI for crosswalk detection
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
            else:
                anti_shake_count = 0

        now = time.time()
        if not parking:
            # 持续设置电机前进（参考 car_race.py 每帧都设置）
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_FORWARD)
            pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
            
            if crosswalk_detected and now - last_trigger_time > COOLDOWN:
                # 检测到斑马线，停车
                pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_STOP)
                parking = True
                park_start_time = now
                play_voice_async()
                print("=== Stopping for {} seconds ===".format(PARK_DELAY))
        else:
            # 停车期间持续保持停止状态
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_STOP)
            pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
            
            elapsed = now - park_start_time
            if elapsed >= PARK_DELAY:
                parking = False
                anti_shake_count = 0
                last_trigger_time = now
                print("=== Resume moving after {:.1f}s ===".format(elapsed))

        # Display every N frames
        if SHOW_WINDOW and frame_count % DISPLAY_EVERY_N == 0:
            cv2.imshow("Zebra Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False

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
