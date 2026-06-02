# -*- coding: utf-8 -*-
"""
=============================================================================
  专业比赛级 双线白色巡线程序
  平台：树莓派 + OpenCV + pigpio
  特性：MJPEG零延迟 | HSV抗反光 | moments双线 | PID控制 | 滑动平均滤波
        丢线保持 | 舵机平滑 | ROI动态 | 安全退出
=============================================================================
"""

import cv2
import numpy as np
import pigpio as pio
import sys
import time
from collections import deque


# ==========================================================================
#  一、硬件初始化（引脚、PWM 参数完全不变）
# ==========================================================================
MOTOR_PIN = 13               # 电机 PWM 引脚（后轮驱动）
STEER_PIN = 12               # 舵机 PWM 引脚（前轮转向）

pi = pio.pi()
if not pi.connected:
    print("❌ pigpio 未连接！请先执行: sudo pigpiod")
    sys.exit(1)

# PWM 频率与量程（与原始代码严格一致）
pi.set_PWM_frequency(MOTOR_PIN, 200)
pi.set_PWM_range(MOTOR_PIN, 40000)        # 电机量程 0-40000
pi.set_PWM_frequency(STEER_PIN, 50)
pi.set_PWM_range(STEER_PIN, 20000)        # 舵机量程 0-20000

# 核心 PWM 参数
SPEED_STOP    = 10000                     # 电机停止
SPEED_FORWARD = 10600                     # 前进速度（稳定不冲）
STEER_CENTER  = 1500                      # 舵机中位（直行）
STEER_MIN     = 1100                      # 舵机左极限
STEER_MAX     = 1900                      # 舵机右极限


# ==========================================================================
#  二、摄像头初始化（V4L2 直驱，绕开 GStreamer）
# ==========================================================================
# ⚠️ cv2.VideoCapture(0) 在某些树莓派上走 GStreamer 后端 → cap.set() 全部失效
#    改用 cv2.CAP_V4L2 直驱 V4L2 → cap.set() 真正生效
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# ↓↓↓ MJPG 压缩 → USB 带宽降低 8 倍，零延迟 ↓↓↓
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 验证实际生效的参数
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fcc_str = "".join([chr((actual_fcc >> i) & 0xFF) for i in (0, 8, 16, 24)])
fcc_str = ''.join(c if c.isprintable() else '?' for c in fcc_str)
print(f"📷 {actual_w}x{actual_h} 编码:{fcc_str}  {'✅ MJPG' if 'MJPG' in fcc_str else '⚠️ 非MJPG(' + fcc_str + ')'}")

FRAME_W = 160   # 画面宽度
FRAME_H = 120   # 画面高度


# ==========================================================================
#  三、ROI 动态区域参数（只看地面，排除墙壁/远处/车头）
# ==========================================================================
ROI_TOP    = 0.50    # ROI 上边界（画面 50%，取底部一半=60px，墙壁在上半部被排除）
ROI_BOTTOM = 0.95    # ROI 下边界（画面 95%，排除车头自身）
ROI_LEFT   = 0.10    # ROI 左边界（排除边缘）
ROI_RIGHT  = 0.90    # ROI 右边界


# ==========================================================================
#  四、视觉识别参数 — 两级阈值检测
# ==========================================================================
# HSV 白色阈值：低饱和度(S≤40) + 中高亮度(V≥160) → 兼顾细线和抗反光
WHITE_LOW  = np.array([0,   0, 160])
WHITE_HIGH = np.array([180, 40, 255])

# 两级阈值：先看全 mask 是否有线，再分半找中心
# 160×120 下 ROI 底部 60×128px，白线 3px 宽 → 180px 总面积
MIN_TOTAL_AREA = 100   # 全 mask 总白像素（判断是否存在线）
MIN_SIDE_AREA  = 30    # 单侧最小像素（决定用哪半算中心，设低防跨边界漏检）


# ==========================================================================
#  五、PID 控制器参数（直道稳、弯道顺）
# ==========================================================================
KP = 3.5        # 比例系数：偏差越大，修正越强
KI = 0.08       # 积分系数：消除直道稳态偏差，帮助回正
KD = 1.5        # 微分系数：抑制弯道过冲与震荡
# ⚠️ 高帧率(>60FPS)时 D 项会被 dt 放大，若弯道震荡可降至 0.3~0.8

PID_INTEGRAL_LIMIT = 300   # 积分限幅，防积分饱和


# ==========================================================================
#  六、滑动平均滤波器 — 消除帧间跳变
# ==========================================================================
FILTER_WINDOW = 5          # 滑动窗口大小（取最近 5 帧平均）


# ==========================================================================
#  七、丢线处理 — 短暂丢线保持动作，不立即停车
# ==========================================================================
LOST_HOLD_FRAMES = 15      # 丢线后保持 15 帧（约 0.5 秒），短暂丢失不立即停车


# ==========================================================================
#  八、舵机平滑参数 — 避免角度突变
# ==========================================================================
STEER_SMOOTH_STEP = 50     # 每帧最大 PWM 变化量


# ==========================================================================
#  八点五、帧降采样 — 周期性抽帧降低 CPU 负载
# ==========================================================================
SKIP_STEP = 3              # 每 N 帧取 1 帧处理，其余仅读丢弃（消费摄像头缓存）


# ==========================================================================
#  显示与调试开关
# ==========================================================================
SHOW_WINDOW = True          # True=调试看画面(VNC会加10-20ms), False=纯控制零延迟
SHOW_EVERY_N = 3            # 每 N 帧显示一次画面，减少 VNC 传输开销
PRINT_EVERY  = 5            # 每 N 帧打印一次（诊断阶段设小点看数据）
DEBUG_VISION = True         # True=每 PRINT_EVERY 帧打印原始白像素数，排查阈值


# ==========================================================================
#  九、全局状态变量
# ==========================================================================
current_steer_pwm = STEER_CENTER    # 当前实际输出 PWM（平滑基准）
lost_counter      = 0              # 连续丢线计数
frame_skip_cnt    = 0              # 帧降采样计数器（全局自增，循环累加）
cx_buffer = deque(maxlen=FILTER_WINDOW)  # cx 滑动窗口

# PID 状态
pid_integral   = 0.0
pid_prev_error = 0.0


# ==========================================================================
#  十、初始化滑动窗口（假设线在画面正中，避免冷启动抖动）
# ==========================================================================
for _ in range(FILTER_WINDOW):
    cx_buffer.append(FRAME_W // 2)


# ==========================================================================
#  十一、视觉识别函数 — 两级阈值混合检测
# ==========================================================================
def detect_white_center(frame):
    """
    两级阈值（~1.5ms @ 树莓派）：
      1) 全 mask 总面积 > MIN_TOTAL_AREA → 有线
      2) 左右分半各取 moments → 双线取中点 / 单线取该侧
      3) 若总面积够但单侧都不够 → 线跨边界 → 用全 mask 质心
    返回 (cx, areaL, areaR) — 诊断时可通过 areaL/areaR 看阈值是否合适
    """
    h, w = frame.shape[:2]
    y1 = int(h * ROI_TOP)
    y2 = int(h * ROI_BOTTOM)
    x1 = int(w * ROI_LEFT)
    x2 = int(w * ROI_RIGHT)
    roi = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 5 or rw < 10:
        return -1, 0, 0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH)

    # 左右分半并取 moments（天然 sum 就是总面积，无需额外 countNonZero）
    mid = rw // 2
    ML = cv2.moments(mask[:, :mid])
    MR = cv2.moments(mask[:, mid:])

    areaL = ML['m00']
    areaR = MR['m00']

    # 第一级：总面积判断是否存在白线
    if areaL + areaR < MIN_TOTAL_AREA:
        return -1, areaL, areaR

    # 第二级：有白线，分半算中心
    has_left  = areaL > MIN_SIDE_AREA
    has_right = areaR > MIN_SIDE_AREA

    if has_left and has_right:
        cxL = int(ML['m10'] / areaL)
        cxR = int(MR['m10'] / areaR) + mid
        raw_cx = (cxL + cxR) // 2
    elif has_left:
        raw_cx = int(ML['m10'] / areaL)
    elif has_right:
        raw_cx = int(MR['m10'] / areaR) + mid
    else:
        # 总面积够但单侧各<30 → 窄线跨边界 → 直接用全 mask 质心
        M = cv2.moments(mask)
        raw_cx = int(M['m10'] / M['m00'])

    return raw_cx + x1, areaL, areaR


# ==========================================================================
#  十二、PID 控制器
# ==========================================================================
def pid_compute(error, dt):
    """
    增量式 PID 控制
      error: 位置偏差 (cx - target)
      dt:    时间间隔 (秒)
      return: PID 输出修正量（PWM 增量）
    """
    global pid_integral, pid_prev_error

    # --- 比例项 ---
    p_out = KP * error

    # --- 积分项（带限幅） ---
    pid_integral += error * dt
    pid_integral = max(-PID_INTEGRAL_LIMIT,
                       min(PID_INTEGRAL_LIMIT, pid_integral))
    i_out = KI * pid_integral

    # --- 微分项（防除零） ---
    d_out = KD * (error - pid_prev_error) / dt if dt > 0 else 0

    pid_prev_error = error
    return p_out + i_out + d_out


# ==========================================================================
#  十三、滑动平均滤波器
# ==========================================================================
def filter_cx(new_cx):
    """对 cx 做滑动平均，消除帧间跳变"""
    cx_buffer.append(new_cx)
    return int(sum(cx_buffer) / len(cx_buffer))


# ==========================================================================
#  十四、舵机平滑输出
# ==========================================================================
def smooth_steer(target_pwm):
    """
    逐步逼近目标 PWM，每帧最多变化 STEER_SMOOTH_STEP
    避免角度突变，保护舵机齿轮
    """
    global current_steer_pwm
    diff = target_pwm - current_steer_pwm

    if abs(diff) <= STEER_SMOOTH_STEP:
        current_steer_pwm = target_pwm
    elif diff > 0:
        current_steer_pwm += STEER_SMOOTH_STEP
    else:
        current_steer_pwm -= STEER_SMOOTH_STEP

    # 硬限幅
    current_steer_pwm = max(STEER_MIN, min(STEER_MAX, current_steer_pwm))
    return int(current_steer_pwm)


# ==========================================================================
#  十五、安全停车 + 舵机回正
# ==========================================================================
def stop_car():
    """电机停止，舵机回正（同时同步全局状态变量）"""
    global current_steer_pwm
    pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_STOP)
    pi.set_PWM_dutycycle(STEER_PIN, STEER_CENTER)
    current_steer_pwm = STEER_CENTER   # 同步状态，避免恢复时角度跳变


# ==========================================================================
#  十六、主控制循环
# ==========================================================================
print("=" * 55)
print("  🏎️  专业比赛级 双线白色巡线")
print("  PID控制 | 滑动平均 | 丢线保持 | 舵机平滑 | 抗反光")
print("  按 'q' 键安全退出")
print("=" * 55)

last_t = time.time()

try:
    while True:
        # --- 帧读取（全量消费摄像头缓存，防止 USB 溢出 / 帧堆积） ---
        ok, frame = cap.read()
        if not ok:
            continue

        # --- 帧降采样：每 SKIP_STEP 帧仅取 1 帧进入巡线流水线 ---
        frame_skip_cnt += 1
        if frame_skip_cnt % SKIP_STEP != 0:
            # 非采样帧：仅消费摄像头缓存，跳过全部图像处理与控制运算
            if SHOW_WINDOW:
                cv2.waitKey(1)   # 保持窗口响应，防止无响应
            continue

        # ============================================================
        #  采样帧：完整巡线流水线
        # ============================================================

        # --- 帧间隔计时（采样帧间真实间隔，供 PID 微分/积分使用） ---
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt <= 0:
            dt = 0.01

        # --- 视觉识别 ---
        cx, aL, aR = detect_white_center(frame)

        # --- 诊断打印（查看原始白像素数，判断阈值是否合适） ---
        diag_cnt = getattr(detect_white_center, '_dc', 0) + 1
        setattr(detect_white_center, '_dc', diag_cnt)
        if DEBUG_VISION and diag_cnt % PRINT_EVERY == 0:
            total = aL + aR
            status = "✅" if cx != -1 else "❌"
            print(f"🔍 白像素 L:{aL:5.0f} R:{aR:5.0f} T:{total:5.0f} "
                  f"(需>{MIN_TOTAL_AREA}) {status}")

        # ============================================================
        #  情况A：找到线 → 正常巡线
        # ============================================================
        if cx != -1:
            # 若刚从较长丢线中恢复（>3帧），重置积分防 windup 过冲
            if lost_counter > 3:
                pid_integral = 0.0
            lost_counter = 0   # 重置丢线计数

            # 1) 滑动平均滤波
            f_cx = filter_cx(cx)

            # 2) 误差计算（画面中心 = 目标位置）
            target = FRAME_W // 2
            error  = f_cx - target

            # 3) PID 计算
            pid_out = pid_compute(error, dt)

            # 4) 目标 PWM + 舵机平滑（限幅保护）
            target_pwm = STEER_CENTER + pid_out
            target_pwm = max(STEER_MIN, min(STEER_MAX, target_pwm))
            steer_pwm  = smooth_steer(int(target_pwm))

            # 5) 执行
            pi.set_PWM_dutycycle(STEER_PIN, steer_pwm)
            pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_FORWARD)

            # 6) 调试输出（限频打印，避免刷屏）
            frame_count = getattr(filter_cx, '_fc', 0) + 1
            setattr(filter_cx, '_fc', frame_count)
            if frame_count % PRINT_EVERY == 0:
                print(f"CX:{cx:3d} | F_CX:{f_cx:3d} | ERR:{error:+4d} | "
                      f"PID:{pid_out:+7.1f} | PWM:{steer_pwm:4d} | ✅")

        # ============================================================
        #  情况B：丢线 → 短暂保持上一帧，超时后停车（仅停车一次）
        # ============================================================
        else:
            lost_counter += 1

            if lost_counter <= LOST_HOLD_FRAMES:
                # 保持当前舵机角度 + 继续前进
                pi.set_PWM_dutycycle(STEER_PIN, int(current_steer_pwm))
                pi.set_PWM_dutycycle(MOTOR_PIN, SPEED_FORWARD)
                # 丢线期间积分自然衰减（每次衰减 10%），防止重新找到线时过冲
                pid_integral *= 0.9
                if lost_counter % 5 == 1:   # 丢线保持也限频打印
                    print(f"⚠️ 丢线保持 [{lost_counter}/{LOST_HOLD_FRAMES}] | "
                          f"PWM:{current_steer_pwm:4d}")
            elif lost_counter == LOST_HOLD_FRAMES + 1:
                # 首次超时：停车一次，清空 PID 状态
                stop_car()
                pid_integral   = 0.0
                pid_prev_error = 0.0
                print(f"🛑 长时间丢线 → 停车")
            # lost_counter > LOST_HOLD_FRAMES+1：已停车，静默等待重新检测到线

        # --- 画面预览（降频显示，减少 VNC 传输开销） ---
        if SHOW_WINDOW:
            dc = getattr(detect_white_center, '_show_cnt', 0) + 1
            setattr(detect_white_center, '_show_cnt', dc)
            if dc % SHOW_EVERY_N == 0:
                cv2.imshow("RaceLine", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n🛑 用户按 'q' 退出")
                break
        else:
            pass

except KeyboardInterrupt:
    print("\n🛑 Ctrl+C 中断")

except Exception as e:
    print(f"\n❌ 运行时异常: {e}")
    import traceback
    traceback.print_exc()

finally:
    # ==================================================================
    #  安全退出：停车 → 舵机回正 → 释放资源
    # ==================================================================
    print("🔧 正在安全退出...")
    stop_car()
    time.sleep(0.2)          # 等待舵机执行回正
    pi.stop()                # 释放 pigpio
    cap.release()            # 释放摄像头
    cv2.destroyAllWindows()  # 关闭窗口
    print("✅ 程序已安全退出，舵机已回正")