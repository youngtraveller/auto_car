# -*- coding: utf-8 -*-
import pigpio as pio
import time
import cv2
import numpy as np
import os
import threading

# ==================== 硬件引脚 ====================
MOTOR_ENA = 13
MOTOR_IN1 = 5
MOTOR_IN2 = 6
STEER_PIN = 12

# ==================== 标准PWM参数（已修复） ====================
MOTOR_FREQ = 100
MOTOR_RANGE = 255
STEER_FREQ = 50
STEER_RANGE = 255

# 电机电平：高电平运转，低电平停止
SPEED_STOP = 0
SPEED_RUN = 200   # 不跑满，降低干扰
STEER_CENTER = 127

# 视觉参数（降低分辨率，减负）
FRAME_WIDTH = 160
FRAME_HEIGHT = 120
VERTICAL_LINE_TOLERANCE = 10
MIN_LINE_LENGTH = 30
MIN_LINES = 3
ROI_BOTTOM_RATIO = 0.6
ANTI_SHAKE_FRAMES = 3

PARK_DELAY = 10
VOICE_PATH = "/home/pi/test_img/voice_module/zebro.wav"
COOLDOWN = 0.5

# ==================== 初始化pigpio ====================
pi = pio.pi()
if not pi.connected:
    exit(1)

# 电机PWM配置
pi.set_PWM_frequency(MOTOR_ENA, MOTOR_FREQ)
pi.set_PWM_range(MOTOR_ENA, MOTOR_RANGE)
# 转向脚设为普通输出
pi.set_mode(MOTOR_IN1, pio.OUTPUT)
pi.set_mode(MOTOR_IN2, pio.OUTPUT)

# 舵机配置
pi.set_PWM_frequency(STEER_PIN, STEER_FREQ)
pi.set_PWM_range(STEER_PIN, STEER_RANGE)

# ==================== 摄像头配置（防帧堆积） ====================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    pi.stop()
    exit(1)

# ==================== 功能函数 ====================
def car_run():
    # 正转方向
    pi.write(MOTOR_IN1, 1)
    pi.write(MOTOR_IN2, 0)
    pi.set_PWM_dutycycle(MOTOR_ENA, SPEED_RUN)
    pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)

def car_stop():
    pi.set_PWM_dutycycle(MOTOR_ENA, SPEED_STOP)
    pi.write(MOTOR_IN1, 0)
    pi.write(MOTOR_IN2, 0)

def play_voice_async():
    try:
        def voice_task():
            os.system(f"aplay {VOICE_PATH} >/dev/null 2>&1")
        threading.Thread(target=voice_task, daemon=True).start()
    except:
        pass

# ==================== 状态变量 ====================
parking = False
park_start_time = 0
anti_shake_count = 0
last_trigger_time = 0
roi_y_start = int(FRAME_HEIGHT * ROI_BOTTOM_RATIO)

# ==================== 主程序 ====================
try:
    car_run()
    print("Car running...")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        roi = frame[roi_y_start:FRAME_HEIGHT, 0:FRAME_WIDTH]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, 60, 10)

        crosswalk_detected = False
        line_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < VERTICAL_LINE_TOLERANCE and \
                   np.sqrt((x2-x1)**2 + (y2-y1)**2) > MIN_LINE_LENGTH:
                    line_count += 1
            if line_count >= MIN_LINES:
                anti_shake_count += 1
                if anti_shake_count >= ANTI_SHAKE_FRAMES:
                    crosswalk_detected = True
            else:
                anti_shake_count = 0

        now = time.time()
        if not parking:
            if crosswalk_detected and now - last_trigger_time > COOLDOWN:
                car_stop()
                parking = True
                park_start_time = now
                play_voice_async()
        else:
            car_stop()
            if now - park_start_time >= PARK_DELAY:
                parking = False
                anti_shake_count = 0
                last_trigger_time = now

        time.sleep(0.01)

finally:
    car_stop()
    cap.release()
    pi.stop()
    print("Exit safely")