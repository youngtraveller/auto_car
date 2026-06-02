# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import threading
from enum import Enum
import RPi.GPIO as GPIO
import logging

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('car.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===================== 全局配置 =====================
class Config:
    # 颜色阈值
    LOWER_BLUE = np.array([100, 100, 50])
    UPPER_BLUE = np.array([130, 255, 255])
    LOWER_RED1 = np.array([0, 120, 50])
    UPPER_RED1 = np.array([10, 255, 255])
    LOWER_RED2 = np.array([170, 120, 50])
    UPPER_RED2 = np.array([180, 255, 255])
    LOWER_YELLOW = np.array([20, 100, 100])
    UPPER_YELLOW = np.array([30, 255, 255])
    
    # 行驶参数
    SAFE_DISTANCE = 0.3
    TRIGGER_DISTANCE = 1.0
    BYPASS_OFFSET = 0.4
    GUIDE_LINE_SPEED = 0.4
    NORMAL_SPEED = 0.5
    BYPASS_SPEED = 0.3
    
    # 坐标转换
    SCALE_X = 0.005
    SCALE_Y = 0.002
    
    # 赛道边界
    TRACK_LEFT_BOUND = -0.8
    TRACK_RIGHT_BOUND = 0.8
    
    # GPIO配置
    ENA = 12
    ENB = 13
    IN1 = 22
    IN2 = 23
    IN3 = 24
    IN4 = 25
    
    # 摄像头配置
    FRAME_WIDTH = 320  # 固定分辨率为320×240以提高帧率
    FRAME_HEIGHT = 240
    TARGET_FPS = 30
    
    # 检测参数
    DETECT_THRESHOLD = 0.5
    COLOR_WEIGHT = 0.5
    SHAPE_WEIGHT = 0.5

# ===================== GPIO配置 =====================
GPIO.setmode(GPIO.BCM)
GPIO.setup([Config.ENA, Config.ENB, Config.IN1, Config.IN2, Config.IN3, Config.IN4], GPIO.OUT)
pwm_a = GPIO.PWM(Config.ENA, 1000)
pwm_b = GPIO.PWM(Config.ENB, 1000)
pwm_a.start(0)
pwm_b.start(0)

# ===================== 状态机 =====================
class CarState(Enum):
    NORMAL = 0
    PREPARE_BYPASS = 1
    LEFT_BYPASS = 2
    RIGHT_BYPASS = 3
    BACK_CENTER = 4
    GUIDE_TRACKING = 5

# ===================== PID控制器 =====================
class PIDSteer:
    def __init__(self, kp=1.2, ki=0.0, kd=0.15):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_err = 0.0
        self.integral = 0.0
        self.lock = threading.Lock()

    def calculate(self, error, dt=0.01):
        with self.lock:
            self.integral += error * dt
            self.integral = max(min(self.integral, 1.0), -1.0)
            diff = (error - self.prev_err) / dt if dt > 0 else 0
            out = self.kp * error + self.ki * self.integral + self.kd * diff
            self.prev_err = error
            return max(min(out, 1.0), -1.0)

# ===================== 性能监控 =====================
class PerformanceMonitor:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.lock = threading.Lock()

    def tick(self):
        with self.lock:
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 1.0:
                self.fps = self.frame_count / elapsed
                logger.info(f"帧率: {self.fps:.1f} FPS")
                self.frame_count = 0
                self.start_time = time.time()
    
    def get_fps(self):
        with self.lock:
            return self.fps

# ===================== 颜色自适应 =====================
def get_adaptive_thresholds(image, detect_yellow=True):
    """根据光照条件动态调整颜色阈值"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = cv2.mean(hsv[:, :, 2])[0]
    
    if detect_yellow:
        lower = Config.LOWER_YELLOW.copy()
        upper = Config.UPPER_YELLOW.copy()
        
        if brightness < 80:  # 低光照
            lower[1] = max(60, lower[1] - 30)
            lower[2] = max(40, lower[2] - 40)
        elif brightness > 200:  # 强光照
            lower[1] = min(150, lower[1] + 30)
            upper[1] = min(255, upper[1] + 20)
        
        return lower, upper
    else:
        return Config.LOWER_BLUE, Config.UPPER_BLUE, Config.LOWER_RED1, Config.UPPER_RED1, Config.LOWER_RED2, Config.UPPER_RED2

# ===================== 形状特征提取 =====================
def extract_shape_features(contour, rect_w, rect_h):
    features = {}
    features['area'] = cv2.contourArea(contour)
    features['perimeter'] = cv2.arcLength(contour, True)
    features['aspect_ratio'] = rect_w / rect_h if rect_h > 0 else 0
    
    if features['perimeter'] > 0:
        features['circularity'] = 4 * np.pi * features['area'] / (features['perimeter'] ** 2)
    else:
        features['circularity'] = 0
    
    features['rectangularity'] = features['area'] / (rect_w * rect_h) if (rect_w * rect_h) > 0 else 0
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    features['solidity'] = features['area'] / hull_area if hull_area > 0 else 0
    
    return features

def calculate_shape_score(features):
    score = 0.0
    weight_sum = 0.0
    
    if 0.3 < features['circularity'] < 0.85:
        score += (features['circularity'] - 0.3) / 0.55 * 0.3
        weight_sum += 0.3
    
    if 0.4 < features['aspect_ratio'] < 1.0:
        score += (1.0 - features['aspect_ratio']) / 0.6 * 0.25
        weight_sum += 0.25
    
    if 0.7 < features['rectangularity'] < 0.95:
        score += (features['rectangularity'] - 0.7) / 0.25 * 0.2
        weight_sum += 0.2
    
    if features['solidity'] > 0.7:
        score += min(features['solidity'], 1.0) * 0.25
        weight_sum += 0.25
    
    return score / weight_sum if weight_sum > 0 else 0.0

def calculate_color_score(mask_roi, total_pixels):
    if total_pixels == 0:
        return 0.0
    color_pixels = cv2.countNonZero(mask_roi)
    return color_pixels / total_pixels

def analyze_spatial_distribution(cones, img_shape):
    """
    分析锥桶的空间分布特征，判断是否符合引导锥桶的分布模式
    
    判定规则（优先级从高到低）：
    1. 数量优先：单个锥桶直接判定为障碍锥桶
    2. 空间分布：多个锥桶需符合引导模式（分布在两侧、纵向排列）
    
    Args:
        cones: 锥桶像素坐标列表 [(x1,y1), (x2,y2), ...]
        img_shape: 图像形状 (height, width)
    
    Returns:
        filtered_cones: 符合引导锥桶分布特征的锥桶列表
        is_guide_pattern: 是否符合引导锥桶分布模式
    """
    # 规则1：单个锥桶直接判定为障碍，不是引导锥桶
    if len(cones) == 1:
        logger.debug(f"单个锥桶，判定为障碍")
        return cones, False
    
    # 规则2：没有锥桶
    if len(cones) == 0:
        return cones, False
    
    h, w = img_shape[:2]
    center_x = w // 2
    
    # 计算分布特征
    x_coords = np.array([c[0] for c in cones])
    y_coords = np.array([c[1] for c in cones])
    
    # 特征1：左右分布情况
    left_cones = [c for c in cones if c[0] < center_x - w * 0.1]
    right_cones = [c for c in cones if c[0] > center_x + w * 0.1]
    
    # 特征2：纵向分布范围（必须有足够的纵向分布）
    y_range = np.max(y_coords) - np.min(y_coords) if len(y_coords) > 1 else 0
    has_longitudinal_distribution = y_range > h * 0.15
    
    # 判断是否符合引导锥桶模式
    is_guide_pattern = False
    if has_longitudinal_distribution:
        # 模式A：两侧都有锥桶（至少各1个）
        if len(left_cones) >= 1 and len(right_cones) >= 1:
            is_guide_pattern = True
            logger.debug(f"模式A：两侧分布，{len(left_cones)}左 + {len(right_cones)}右")
        # 模式B：单侧有多个锥桶且纵向分布明显
        elif (len(left_cones) >= 3 or len(right_cones) >= 3) and y_range > h * 0.25:
            is_guide_pattern = True
            logger.debug(f"模式B：单侧密集分布，数量={max(len(left_cones), len(right_cones))}")
    
    # 过滤掉孤立的中心锥桶（可能是障碍锥桶误检）
    if is_guide_pattern:
        filtered_cones = left_cones + right_cones
    else:
        # 如果不符合引导模式，保留所有锥桶（作为障碍锥桶）
        filtered_cones = cones
    
    return filtered_cones, is_guide_pattern

# ===================== 视觉检测线程 =====================
class VisionProcessor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.cap = None
        self.running = False
        self.latest_frame = None
        self.barrier_cones = []
        self.guide_cones = []
        self.lock = threading.Lock()
        self.monitor = PerformanceMonitor()

    def run(self):
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, Config.TARGET_FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info("摄像头初始化成功")
            
            self.running = True
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                # 自适应颜色阈值
                yellow_lower, yellow_upper = get_adaptive_thresholds(frame, detect_yellow=True)
                
                # 检测黄色锥桶
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
                yellow_candidates = self.detect_cones(frame, mask_yellow)
                
                # 分析空间分布，判断是否为引导锥桶
                guide_cones, is_guide_pattern = analyze_spatial_distribution(yellow_candidates, frame.shape)
                
                if is_guide_pattern:
                    logger.debug(f"检测到引导锥桶模式: {len(guide_cones)}个锥桶")
                else:
                    logger.debug(f"黄色锥桶不符合引导模式，作为障碍处理: {len(yellow_candidates)}个")
                
                # 检测障碍锥桶（红/蓝）
                blue_lower, blue_upper, red1_lower, red1_upper, red2_lower, red2_upper = get_adaptive_thresholds(frame, detect_yellow=False)
                mask1 = cv2.inRange(hsv, blue_lower, blue_upper)
                mask2 = cv2.inRange(hsv, red1_lower, red1_upper)
                mask3 = cv2.inRange(hsv, red2_lower, red2_upper)
                mask_barrier = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
                barrier_cones = self.detect_cones(frame, mask_barrier)
                
                # 如果黄色锥桶不符合引导模式，加入障碍列表
                if not is_guide_pattern and len(yellow_candidates) > 0:
                    barrier_cones.extend(yellow_candidates)
                    guide_cones = []
                
                # 更新检测结果
                with self.lock:
                    self.latest_frame = frame.copy()
                    self.guide_cones = guide_cones
                    self.barrier_cones = barrier_cones
                
                self.monitor.tick()
                
        except Exception as e:
            logger.error(f"视觉处理线程异常: {e}")
        finally:
            if self.cap:
                self.cap.release()

    def detect_cones(self, image, mask):
        cone_positions = []
        
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = image.shape[:2]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            min_area = max(40, h * w * 0.0002)
            max_area = h * w * 0.08
            if area < min_area or area > max_area:
                continue
            
            x, y, rect_w, rect_h = cv2.boundingRect(cnt)
            min_height = max(10, h * 0.025)
            min_width = max(8, w * 0.015)
            if rect_h < min_height or rect_w < min_width:
                continue
            
            shape_features = extract_shape_features(cnt, rect_w, rect_h)
            shape_score = calculate_shape_score(shape_features)
            
            mask_roi = mask[y:y+rect_h, x:x+rect_w]
            total_pixels = rect_w * rect_h
            color_score = calculate_color_score(mask_roi, total_pixels)
            
            final_score = Config.COLOR_WEIGHT * color_score + Config.SHAPE_WEIGHT * shape_score
            
            if final_score > Config.DETECT_THRESHOLD:
                cx = x + rect_w // 2
                cy = y + rect_h // 2
                cone_positions.append((int(cx), int(cy)))
        
        return cone_positions

    def get_results(self):
        with self.lock:
            return self.latest_frame, self.barrier_cones.copy(), self.guide_cones.copy()

    def stop(self):
        self.running = False
        self.join(timeout=1.0)

# ===================== 电机控制 =====================
def set_motor_speed(left_speed, right_speed):
    left_speed = max(min(left_speed, 100), -100)
    right_speed = max(min(right_speed, 100), -100)
    
    if left_speed >= 0:
        GPIO.output(Config.IN1, GPIO.HIGH)
        GPIO.output(Config.IN2, GPIO.LOW)
        pwm_a.ChangeDutyCycle(left_speed)
    else:
        GPIO.output(Config.IN1, GPIO.LOW)
        GPIO.output(Config.IN2, GPIO.HIGH)
        pwm_a.ChangeDutyCycle(-left_speed)
    
    if right_speed >= 0:
        GPIO.output(Config.IN3, GPIO.HIGH)
        GPIO.output(Config.IN4, GPIO.LOW)
        pwm_b.ChangeDutyCycle(right_speed)
    else:
        GPIO.output(Config.IN3, GPIO.LOW)
        GPIO.output(Config.IN4, GPIO.HIGH)
        pwm_b.ChangeDutyCycle(-right_speed)

def stop_motors():
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)

# ===================== 坐标转换和路径规划 =====================
def pixel2car(pixel_pos, img_shape):
    if not pixel_pos:
        return None
    h, w = img_shape[:2]
    px, py = pixel_pos
    car_y = (px - w // 2) * Config.SCALE_Y
    car_x = (h - py) * Config.SCALE_X
    return (car_x, car_y)

def predict_path_direction(cones_pixel, img_shape, look_ahead_distance=30):
    """
    根据多个引导锥桶预测路径方向
    
    Args:
        cones_pixel: 锥桶像素坐标列表 [(x1,y1), (x2,y2), ...]
        img_shape: 图像形状 (height, width)
        look_ahead_distance: 前瞻距离（像素）
    
    Returns:
        target_x: 预测的目标X坐标（像素）
        curvature: 路径曲率（正为左转，负为右转）
        confidence: 预测置信度 (0-1)
    """
    if len(cones_pixel) < 2:
        if len(cones_pixel) == 1:
            return cones_pixel[0][0], 0, 0.5
        return img_shape[1] // 2, 0, 0.0
    
    # 按Y坐标排序（从远到近）
    sorted_cones = sorted(cones_pixel, key=lambda c: c[1])
    
    # 使用前N个锥桶进行预测
    max_cones = min(5, len(sorted_cones))
    selected_cones = sorted_cones[:max_cones]
    
    # 提取坐标
    x_coords = np.array([c[0] for c in selected_cones], dtype=np.float32)
    y_coords = np.array([c[1] for c in selected_cones], dtype=np.float32)
    
    # 计算相对距离（越近权重越大）
    weights = np.linspace(0.3, 1.0, len(selected_cones))
    
    try:
        # 使用加权最小二乘法拟合二次曲线 y = ax² + bx + c
        # 转换为线性方程组求解
        n = len(selected_cones)
        X = np.column_stack([y_coords**2, y_coords, np.ones(n)])
        
        # 加权拟合
        W = np.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ x_coords
        
        # 求解
        coeffs = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
        a, b, c = coeffs
        
        # 计算当前位置的路径方向（切线斜率）
        current_y = img_shape[0] * 0.6  # 假设车辆在画面下方60%位置
        slope = 2 * a * current_y + b
        
        # 计算前瞻位置的目标X坐标
        look_ahead_y = current_y - look_ahead_distance
        if look_ahead_y < 0:
            look_ahead_y = 0
        
        target_x_predicted = a * look_ahead_y**2 + b * look_ahead_y + c
        
        # 计算曲率（二次导数）
        curvature = 2 * a * Config.SCALE_Y / (1 + slope**2)**1.5
        
        # 计算置信度（基于拟合误差和锥桶数量）
        predictions = X @ coeffs
        rmse = np.sqrt(np.mean((x_coords - predictions)**2))
        fit_confidence = max(0, 1 - rmse / (img_shape[1] * 0.1))
        count_confidence = min(1.0, len(selected_cones) / 5.0)
        confidence = (fit_confidence + count_confidence) / 2
        
        return target_x_predicted, curvature, confidence
        
    except Exception as e:
        logger.warning(f"路径预测失败: {e}")
        # 降级到简单平均
        avg_x = np.mean(x_coords)
        return avg_x, 0, 0.5

def get_guide_path_center(cones_pixel, img_shape):
    if len(cones_pixel) < 1:
        return 0.0
    
    # 使用路径预测
    target_x, curvature, confidence = predict_path_direction(cones_pixel, img_shape)
    
    # 根据置信度混合预测结果和简单平均
    if confidence > 0.6:
        # 高置信度，使用预测结果
        final_x = target_x
    else:
        # 低置信度，使用简单平均
        x_coords = [x for x, y in cones_pixel]
        final_x = np.mean(x_coords)
    
    img_w = img_shape[1]
    target_y = (final_x - img_w // 2) * Config.SCALE_Y
    
    # 根据曲率调整目标位置，提前转向
    curvature_factor = curvature * 0.5  # 曲率影响系数
    target_y += curvature_factor
    
    return max(min(target_y, Config.TRACK_RIGHT_BOUND - Config.SAFE_DISTANCE), 
               Config.TRACK_LEFT_BOUND + Config.SAFE_DISTANCE)

def prioritize_obstacles(cones, img_shape):
    """
    按威胁程度排序障碍物
    
    Args:
        cones: 锥桶像素坐标列表
        img_shape: 图像形状
    
    Returns:
        按距离排序的锥桶列表（近的在前）
    """
    obstacles = []
    for cone in cones:
        cone_car = pixel2car(cone, img_shape)
        if cone_car:
            distance = cone_car[0]
            obstacles.append((distance, cone))
    
    # 按距离排序（近的优先）
    obstacles.sort(key=lambda x: x[0])
    return [cone for _, cone in obstacles]

def get_adaptive_safe_distance(speed):
    """
    根据速度动态调整安全距离
    
    Args:
        speed: 当前车速
    
    Returns:
        自适应安全距离
    """
    base_distance = Config.SAFE_DISTANCE
    # 速度越快，安全距离越大
    return base_distance + speed * 0.15

def choose_bypass_dir(cone_car_pos, current_speed=0.3):
    """
    选择绕行方向（优化版）
    
    Args:
        cone_car_pos: 障碍物在车辆坐标系中的位置
        current_speed: 当前车速（用于动态调整）
    
    Returns:
        (target_y, direction) 或 (None, None)
    """
    if not cone_car_pos:
        return None, None
    
    _, cone_y = cone_car_pos
    adaptive_safe = get_adaptive_safe_distance(current_speed)
    
    space_left = cone_y - Config.TRACK_LEFT_BOUND
    space_right = Config.TRACK_RIGHT_BOUND - cone_y
    
    # 根据空间大小选择绕行方向
    if space_left > space_right + 0.1:  # 左侧空间明显更大
        if space_left > adaptive_safe:
            # 根据速度调整偏移量
            dynamic_offset = Config.BYPASS_OFFSET * (1 + 0.3 * min(current_speed / Config.NORMAL_SPEED, 1))
            target_y = max(cone_y - dynamic_offset, Config.TRACK_LEFT_BOUND + adaptive_safe)
            logger.debug(f"选择左侧绕行，空间: {space_left:.2f}m，目标: {target_y:.2f}")
            return target_y, "left"
    elif space_right > space_left + 0.1:  # 右侧空间明显更大
        if space_right > adaptive_safe:
            dynamic_offset = Config.BYPASS_OFFSET * (1 + 0.3 * min(current_speed / Config.NORMAL_SPEED, 1))
            target_y = min(cone_y + dynamic_offset, Config.TRACK_RIGHT_BOUND - adaptive_safe)
            logger.debug(f"选择右侧绕行，空间: {space_right:.2f}m，目标: {target_y:.2f}")
            return target_y, "right"
    else:  # 两侧空间相近，选择障碍物较小的一侧
        if space_left > adaptive_safe:
            target_y = max(cone_y - Config.BYPASS_OFFSET, Config.TRACK_LEFT_BOUND + adaptive_safe)
            return target_y, "left"
        elif space_right > adaptive_safe:
            target_y = min(cone_y + Config.BYPASS_OFFSET, Config.TRACK_RIGHT_BOUND - adaptive_safe)
            return target_y, "right"
    
    return None, None

# ===================== 主程序 =====================
def main():
    logger.info("🚗 无人车程序启动")
    
    # 启动视觉处理线程
    vision_processor = VisionProcessor()
    vision_processor.start()
    time.sleep(1.0)  # 等待摄像头初始化
    
    pid = PIDSteer()
    now_state = CarState.NORMAL
    target_y = 0.0
    car_current_y = 0.0
    last_time = time.time()
    bypass_timer = 0.0  # 绕行计时器
    
    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # 获取检测结果
            frame, barrier_cones, guide_cones = vision_processor.get_results()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            img_h, img_w = frame.shape[:2]
            
            # 当前速度（用于动态调整）
            current_speed = Config.NORMAL_SPEED
            if now_state == CarState.GUIDE_TRACKING:
                current_speed = Config.GUIDE_LINE_SPEED
            elif now_state in [CarState.LEFT_BYPASS, CarState.RIGHT_BYPASS]:
                current_speed = Config.BYPASS_SPEED
            
            # 状态机逻辑
            try:
                if now_state == CarState.NORMAL:
                    target_y = 0.0
                    if len(barrier_cones) > 0:
                        # 多障碍物优先级处理
                        prioritized_cones = prioritize_obstacles(barrier_cones, frame.shape)
                        if prioritized_cones:
                            cone_car = pixel2car(prioritized_cones[0], frame.shape)
                            if cone_car:
                                distance = cone_car[0]
                                # 根据速度动态调整触发距离
                                adaptive_trigger = Config.TRIGGER_DISTANCE + current_speed * 0.2
                                if distance < adaptive_trigger:
                                    now_state = CarState.PREPARE_BYPASS
                                    logger.info(f"检测到障碍物，距离: {distance:.2f}m，准备绕行")
                    elif len(guide_cones) > 0:
                        now_state = CarState.GUIDE_TRACKING
                        logger.info(f"进入引导跟踪模式")

                elif now_state == CarState.PREPARE_BYPASS:
                    if len(barrier_cones) > 0:
                        # 使用优先级最高的障碍物
                        prioritized_cones = prioritize_obstacles(barrier_cones, frame.shape)
                        if prioritized_cones:
                            cone_car = pixel2car(prioritized_cones[0], frame.shape)
                            res_y, direction = choose_bypass_dir(cone_car, current_speed)
                            if direction == "left":
                                target_y = res_y
                                now_state = CarState.LEFT_BYPASS
                                bypass_timer = 0.0
                                logger.info(f"向左绕行，目标位置: {target_y:.2f}")
                            elif direction == "right":
                                target_y = res_y
                                now_state = CarState.RIGHT_BYPASS
                                bypass_timer = 0.0
                                logger.info(f"向右绕行，目标位置: {target_y:.2f}")
                            else:
                                # 无法绕行，减速并保持直行
                                logger.warning("无法绕行，减速通过")
                                now_state = CarState.NORMAL
                    else:
                        now_state = CarState.NORMAL

                elif now_state in [CarState.LEFT_BYPASS, CarState.RIGHT_BYPASS]:
                    bypass_timer += dt
                    
                    # 优先检测引导锥桶
                    if len(guide_cones) > 0:
                        now_state = CarState.GUIDE_TRACKING
                        logger.info(f"检测到引导锥桶，切换跟踪模式")
                    # 检查障碍物是否已通过
                    elif len(barrier_cones) == 0:
                        now_state = CarState.BACK_CENTER
                        logger.info("障碍物已通过，返回中心")
                    # 绕行超时保护
                    elif bypass_timer > 5.0:  # 最多绕行5秒
                        logger.warning("绕行超时，强制返回中心")
                        now_state = CarState.BACK_CENTER

                elif now_state == CarState.BACK_CENTER:
                    target_y = 0.0
                    if len(guide_cones) > 0:
                        now_state = CarState.GUIDE_TRACKING
                    elif abs(car_current_y - target_y) < 0.05:
                        now_state = CarState.NORMAL
                        logger.info("已返回赛道中心")

                elif now_state == CarState.GUIDE_TRACKING:
                    # 引导模式下也检测障碍物
                    if len(barrier_cones) > 0:
                        prioritized_cones = prioritize_obstacles(barrier_cones, frame.shape)
                        if prioritized_cones:
                            cone_car = pixel2car(prioritized_cones[0], frame.shape)
                            if cone_car:
                                distance = cone_car[0]
                                adaptive_trigger = Config.TRIGGER_DISTANCE + current_speed * 0.15
                                if distance < adaptive_trigger:
                                    now_state = CarState.PREPARE_BYPASS
                                    logger.info(f"引导模式下检测到障碍物，距离: {distance:.2f}m")
                    
                    if len(guide_cones) == 0:
                        target_y = 0.0
                        now_state = CarState.NORMAL
                        logger.info("引导锥桶丢失，返回正常模式")
                    else:
                        target_y = get_guide_path_center(guide_cones, frame.shape)
            
            except Exception as e:
                logger.error(f"状态机异常: {e}")
                now_state = CarState.NORMAL
            
            # PID控制
            offset_err = target_y - car_current_y
            steer_ctrl = pid.calculate(offset_err, dt)
            
            # 速度控制
            if now_state == CarState.GUIDE_TRACKING:
                run_speed = Config.GUIDE_LINE_SPEED
            elif now_state in [CarState.LEFT_BYPASS, CarState.RIGHT_BYPASS]:
                run_speed = Config.BYPASS_SPEED
            else:
                run_speed = Config.NORMAL_SPEED
            
            # 转换为电机速度
            left_speed = (1 - steer_ctrl) * run_speed * 100
            right_speed = (1 + steer_ctrl) * run_speed * 100
            
            # 控制电机
            set_motor_speed(left_speed, right_speed)
            
            # 更新当前位置（简化模型）
            car_current_y += steer_ctrl * 0.02
            
            # 可视化
            for (x, y) in barrier_cones:
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
            for (x, y) in guide_cones:
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(frame, f"State: {now_state.name}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"FPS: {vision_processor.monitor.get_fps():.1f}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Target: {target_y:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Race Car", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("收到退出信号")
                break
            
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"主程序异常: {e}")
    finally:
        vision_processor.stop()
        stop_motors()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        logger.info("🏁 程序退出")

if __name__ == "__main__":
    main()
