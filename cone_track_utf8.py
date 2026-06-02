# -*- coding: utf-8 -*-
import cv2
import numpy as np
import serial
import time
from enum import Enum

# ===================== 颜色阈值参数 =====================
LOWER_BLUE = np.array([100, 100, 50])
UPPER_BLUE = np.array([130, 255, 255])
LOWER_RED1 = np.array([0, 120, 50])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 120, 50])
UPPER_RED2 = np.array([180, 255, 255])
LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([30, 255, 255])

# ===================== 控制参数 =====================
SAFE_DISTANCE = 0.3
TRIGGER_DISTANCE = 1.0
BYPASS_OFFSET = 0.4
GUIDE_LINE_SPEED = 0.4

SCALE_X = 0.005   # 像素转X距离比例
SCALE_Y = 0.002   # 像素转Y偏移比例

TRACK_LEFT_BOUND = -0.8
TRACK_RIGHT_BOUND = 0.8

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 9600
SIMULATION_MODE = True  # 仿真模式：不发送串口指令

# ===================== 小车状态机 =====================
class CarState(Enum):
    NORMAL = 0          # 正常直行
    PREPARE_BYPASS = 1  # 准备避障
    LEFT_BYPASS = 2     # 左侧绕行
    RIGHT_BYPASS = 3    # 右侧绕行
    BACK_CENTER = 4     # 返回赛道中心
    GUIDE_TRACKING = 5  # 黄色锥桶引导循迹

# ===================== PID 转向控制器 =====================
class PIDSteer:
    def __init__(self, kp=1.2, ki=0.0, kd=0.15):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_err = 0.0
        self.integral = 0.0

    def calculate(self, error, dt=0.01):
        self.integral += error * dt
        diff = (error - self.prev_err) / dt
        out = self.kp * error + self.ki * self.integral + self.kd * diff
        self.prev_err = error
        # 限幅 -1 ~ 1
        return max(min(out, 1.0), -1.0)

# ===================== 串口通信 =====================
class CarSerial:
    def __init__(self, port, baud):
        self.ser = None
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            time.sleep(2)
            print("? 串口连接成功")
        except Exception as e:
            print("? 串口连接失败:", e)

    def send_control(self, steer, speed):
        if self.ser and self.ser.is_open:
            cmd = f"{steer:.2f},{speed:.2f}\r\n"
            self.ser.write(cmd.encode('utf-8'))

    def close(self):
        if self.ser:
            self.ser.close()

# ===================== 视觉检测 =====================
def detect_all_cones(image, detect_yellow=True):
    """检测锥桶：黄色=引导，红蓝=障碍物"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cone_positions = []
    
    if detect_yellow:
        mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    else:
        mask1 = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
        mask2 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
        mask3 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
        mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))

    # 形态学去噪
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:  # 降低最小面积，适配你的小锥桶
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        circularity = area / (np.pi * r**2) if r > 0 else 0
        if circularity > 0.4:  # 降低圆形度要求，更容易识别
            cone_positions.append((int(x), int(y)))
    return cone_positions

def detect_track_line(image):
    """检测赛道边线（备用）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    edge = cv2.Canny(bin_img, 50, 150)
    lines = cv2.HoughLinesP(edge, 1, np.pi/180, 50, minLineLength=50, maxLineGap=20)
    left_line = right_line = None
    if lines is not None:
        left_group, right_group = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.3:
                left_group.append(line[0])
            elif slope > 0.3:
                right_group.append(line[0])
        if left_group:
            left_line = np.mean(left_group, axis=0).astype(int)
        if right_group:
            right_line = np.mean(right_group, axis=0).astype(int)
    return left_line, right_line

def pixel2car(pixel_pos, img_shape):
    """
    像素坐标 → 小车坐标系
    返回 (前方距离x, 左右偏移y)
    """
    if not pixel_pos:
        return None
    h, w = img_shape[:2]
    px, py = pixel_pos
    # 左右偏移：中心为0
    car_y = (px - w // 2) * SCALE_Y
    # 前后距离：越近值越小
    car_x = (h - py) * SCALE_X
    return (car_x, car_y)

# ===================== 路径规划 =====================
def get_guide_path_center(cones_pixel, img_shape):
    """计算黄色引导锥桶的中心目标点"""
    if len(cones_pixel) < 1:
        return 0.0
    x_coords = [x for x, y in cones_pixel]
    target_x = np.mean(x_coords)
    img_w = img_shape[1]
    target_y = (target_x - img_w // 2) * SCALE_Y
    # 安全边界限制
    return max(min(target_y, TRACK_RIGHT_BOUND - SAFE_DISTANCE), TRACK_LEFT_BOUND + SAFE_DISTANCE)

def choose_bypass_dir(cone_car_pos):
    """选择避障方向：左/右"""
    if not cone_car_pos:
        return None, None
    _, cone_y = cone_car_pos
    space_left = cone_y - TRACK_LEFT_BOUND
    space_right = TRACK_RIGHT_BOUND - cone_y
    
    if space_left > space_right and space_left > SAFE_DISTANCE:
        return max(cone_y - BYPASS_OFFSET, TRACK_LEFT_BOUND + SAFE_DISTANCE), "left"
    elif space_right > space_left and space_right > SAFE_DISTANCE:
        return min(cone_y + BYPASS_OFFSET, TRACK_RIGHT_BOUND - SAFE_DISTANCE), "right"
    return None, None

# ===================== 主函数 =====================
def main():
    # 摄像头初始化
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    pid = PIDSteer()
    serial_ctrl = CarSerial(SERIAL_PORT, BAUD_RATE)
    now_state = CarState.NORMAL
    target_y = 0.0        # 目标左右位置
    car_current_y = 0.0   # 小车当前位置（仿真用）

    print("?? 程序启动，按 Q 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("? 摄像头读取失败")
            break
        
        img_h, img_w = frame.shape[:2]

        # 视觉检测
        barrier_cones = detect_all_cones(frame, detect_yellow=False)  # 障碍物锥桶
        guide_cones = detect_all_cones(frame, detect_yellow=True)     # 引导锥桶
        left_line, right_line = detect_track_line(frame)              # 边线

        # 调试输出
        print(f"[状态] {now_state.name:<15} | 障碍物:{len(barrier_cones)}  引导:{len(guide_cones)}")

        # ===================== 状态机逻辑 =====================
        if now_state == CarState.NORMAL:
            target_y = 0.0  # 车道中心
            # 检测到近距离障碍物 → 准备避障
            if len(barrier_cones) > 0:
                cone_car = pixel2car(barrier_cones[0], frame.shape)
                if cone_car is not None and cone_car[0] < TRIGGER_DISTANCE:
                    print("??  检测到障碍物，准备避障")
                    now_state = CarState.PREPARE_BYPASS
            # 检测到引导锥桶 → 循迹
            elif len(guide_cones) > 0:
                print("?? 检测到引导锥桶，进入循迹模式")
                now_state = CarState.GUIDE_TRACKING

        elif now_state == CarState.PREPARE_BYPASS:
            if len(barrier_cones) == 0:
                now_state = CarState.NORMAL
                print("?? 障碍物消失，返回正常模式")
                continue
            cone_car = pixel2car(barrier_cones[0], frame.shape)
            res_y, direction = choose_bypass_dir(cone_car)
            if direction == "left":
                target_y = res_y
                now_state = CarState.LEFT_BYPASS
                print("??  开始左侧避障")
            elif direction == "right":
                target_y = res_y
                now_state = CarState.RIGHT_BYPASS
                print("??  开始右侧避障")
            else:
                now_state = CarState.NORMAL
                print("??  无安全空间，返回正常模式")

        elif now_state in [CarState.LEFT_BYPASS, CarState.RIGHT_BYPASS]:
            # 避障中发现引导锥桶 → 切换循迹
            if len(guide_cones) > 0:
                now_state = CarState.GUIDE_TRACKING
                print("?? 切换到引导循迹")
            # 障碍物消失 → 准备回中心
            elif len(barrier_cones) == 0:
                now_state = CarState.BACK_CENTER
                print("?? 避障完成，准备返回车道中心")

        elif now_state == CarState.BACK_CENTER:
            target_y = 0.0
            if len(guide_cones) > 0:
                now_state = CarState.GUIDE_TRACKING
                print("?? 切换到引导循迹")
            # 回到中心 → 正常行驶
            elif abs(car_current_y - target_y) < 0.05:
                now_state = CarState.NORMAL
                print("? 已回到车道中心")

        elif now_state == CarState.GUIDE_TRACKING:
            # 引导锥桶消失 → 回到中心正常行驶
            if len(guide_cones) == 0:
                target_y = 0.0
                now_state = CarState.NORMAL
                print("?? 引导结束，恢复正常行驶")
            else:
                target_y = get_guide_path_center(guide_cones, frame.shape)

        # ===================== PID 控制 =====================
        offset_err = target_y - car_current_y
        steer_ctrl = pid.calculate(offset_err)
        
        # 模拟小车位置更新
        car_current_y += steer_ctrl * 0.01
        
        # 速度控制
        if now_state == CarState.GUIDE_TRACKING:
            run_speed = GUIDE_LINE_SPEED
        elif now_state in [CarState.LEFT_BYPASS, CarState.RIGHT_BYPASS]:
            run_speed = 0.3  # 避障减速
        else:
            run_speed = 0.5  # 正常速度

        # 发送控制指令
        if not SIMULATION_MODE:
            serial_ctrl.send_control(steer_ctrl, run_speed)

        # ===================== 画面绘制 =====================
        # 绘制障碍物锥桶（红）
        for (x, y) in barrier_cones:
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        # 绘制引导锥桶（黄）
        for (x, y) in guide_cones:
            cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
        
        # 显示状态
        cv2.putText(frame, f"State: {now_state.name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imshow("Race Car Vision", frame)

        # 退出条件
        key = cv2.waitKey(1)
        if key != -1:
           if key & 0xFF == ord('q'):
             break
          

    # 释放资源
    cap.release()
    serial_ctrl.close()
    cv2.destroyAllWindows()
    print("?? 程序已退出")

if __name__ == "__main__":
    main()