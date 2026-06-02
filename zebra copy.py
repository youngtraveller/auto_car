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

# Speed settings (same as car_race.py)
SPEED_STOP = 10000
SPEED_RUN = 10600
STEER_CENTER = 1500

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

# Display settings - 抽帧显示降低延迟
SHOW_WINDOW = True
DISPLAY_EVERY_N = 5  # 每5帧显示一次，降低VNC传输开销

# Control functions
def set_motor(speed):
    pi.set_PWM_dutycycle(MOTOR_PIN, speed)

def set_steer(angle):
    pi.set_PWM_dutycycle(STEER_PIN, angle)

def keep_run():
    set_steer(STEER_CENTER)
    set_motor(SPEED_RUN)

def car_stop():
    set_motor(SPEED_STOP)
    set_steer(STEER_CENTER)

def play_voice_async():
    def voice_thread():
        os.system("aplay {}".format(VOICE_PATH))
    threading.Thread(target=voice_thread, daemon=True).start()

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
print("Display: every {} frames".format(DISPLAY_EVERY_N))
keep_run()
print("Car started moving")

try:
    while running:
        frame_count += 1
        
        # Read frame
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        
        # Process ROI for crosswalk detection (每帧都运行)
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
            keep_run()
            if crosswalk_detected and now - last_trigger_time > COOLDOWN:
                car_stop()
                parking = True
                park_start_time = now
                play_voice_async()
                print("=== Stopping for {} seconds ===".format(PARK_DELAY))
        else:
            car_stop()
            elapsed = now - park_start_time
            if elapsed >= PARK_DELAY:
                parking = False
                anti_shake_count = 0
                last_trigger_time = now
                print("=== Resume moving after {:.1f}s ===".format(elapsed))

        # 抽帧显示 - 只有每N帧才显示画面
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
    car_stop()
    time.sleep(0.2)
    pi.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Done")