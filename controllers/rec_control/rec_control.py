"""
================================================
  YouBot Q-Learning Controller  —  النسخة المُصلحة
================================================
الإصلاحات المطبّقة:
  1. حدود البيئة صُحّحت: X:±5.5 / Z:±3.5
     (الحد القديم ±2.0 كان يعاقب الروبوت قبل وصوله للهدف)

  2. شرط goal_reached أصبح دقيقاً:
     يشترط رؤية الهدف الأخضر + مساحته >= 400 بكسل + في المنتصف
     + حساس المسافة < 300 (كلها معاً لمنع الإيجابيات الكاذبة)

  3. تسلسل الإمساك متعدد المراحل مع انتظار استقرار فعلي:
     PICKING_APPROACH → PICKING_CLOSE → PICKING_LIFT → FINISHED

  4. قيم الثواب أُعيد توازنها (وصول: 200، اقتراب: 20، خروج: -50)

  5. دالة green_target تُنقي الضوضاء بعملية مورفولوجية

  6. تحميل النموذج المحفوظ تلقائياً عند البدء

  7. حساسات موضع الذراع مُفعَّلة للتحقق من الاستقرار
================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from controller import Robot, Motor, GPS, DistanceSensor, Camera, InertialUnit


# ─────────────────────────────────────────────
# 1.  شبكة Q العصبية
# ─────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────────
# 2.  تهيئة الروبوت والأجهزة
# ─────────────────────────────────────────────
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ── عجلات ──
wheels = []
for i in range(1, 5):
    w = robot.getDevice(f'wheel{i}')
    w.setPosition(float('inf'))
    w.setVelocity(0.0)
    wheels.append(w)

# ── مفاصل الذراع والأصابع ──
arm_joints = [robot.getDevice(f'arm{i}') for i in range(1, 6)]
left_finger = robot.getDevice('finger::left')
right_finger = robot.getDevice('finger::right')

# ── حساسات موضع الذراع (ضرورية لمتابعة الاستقرار) ──
arm_sensors = []
for i in range(1, 6):
    ps = robot.getDevice(f'arm{i}sensor')
    ps.enable(timestep)
    arm_sensors.append(ps)

left_finger_sensor = robot.getDevice('finger::leftsensor')
right_finger_sensor = robot.getDevice('finger::rightsensor')
left_finger_sensor.enable(timestep)
right_finger_sensor.enable(timestep)

# ── حساسات البيئة ──
gps = robot.getDevice('gps')
gps.enable(timestep)

ds_front = robot.getDevice('ds_front')
ds_front.enable(timestep)

imu = robot.getDevice('inertial unit')
imu.enable(timestep)

camera = robot.getDevice('camera')
camera.enable(timestep)

# ─────────────────────────────────────────────
# 3.  النموذج والمُحسِّن
# ─────────────────────────────────────────────
model = QNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

MODEL_FILE = "youbot_q_network.pth"

# تحميل نموذج محفوظ مسبقاً إن وُجد
if os.path.exists(MODEL_FILE):
    try:
        model.load_state_dict(torch.load(MODEL_FILE, map_location='cpu'))
        print(f"[INFO] تم تحميل النموذج المحفوظ من: {MODEL_FILE}")
    except Exception as e:
        print(f"[WARN] تعذّر تحميل النموذج ({e}) — سيبدأ من الصفر")

# ─────────────────────────────────────────────
# 4.  ثوابت التدريب والبيئة
# ─────────────────────────────────────────────
ROBOT_STATE = "TRAINING"
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
gamma = 0.95

directions = ["FORWARD", "BACKWARD", "TURN_RIGHT", "TURN_LEFT", "STOP"]

# ── حدود البيئة المُصلَّحة ──
# الصناديق الخضراء موجودة حتى X=4.75 وZ=-2.93
# الحد القديم ±2.0 كان يعاقب الروبوت قبل وصوله للهدف
ENV_LIMIT_X = 5.5
ENV_LIMIT_Z = 3.5

# ── إعدادات الكشف الأخضر (HSV) ──
GREEN_LOW = np.array([35, 60, 60], dtype=np.uint8)
GREEN_HIGH = np.array([85, 255, 255], dtype=np.uint8)

# ── متغيرات الحالة ──
last_seen_area = 0
consecutive_stuck = 0

# ── إعدادات الإمساك ──
PICK_ARM2_DOWN = -1.13  # إنزال الكتف
PICK_ARM3_DOWN = -1.36  # انحناء الكوع للأمام
PICK_FINGER_OPEN = 0.025  # فتح الأصابع
PICK_FINGER_CLOSED = 0.006  # إغلاق الأصابع (للجسم 10سم)
LIFT_ARM2_UP = -0.5  # رفع الكتف
LIFT_ARM3_UP = -0.5  # رفع الكوع

# ── متغيرات تسلسل الإمساك ──
pick_start_time = 0.0
pick_phase = 0


# ─────────────────────────────────────────────
# 5.  دوال مساعدة
# ─────────────────────────────────────────────

def green_target():
    """
    يُحلّل إطار الكاميرا ويكشف الجسم الأخضر.
    يُعيد: (target_dir, area, cx_norm)
      target_dir : -1=يسار، 0=وسط، 1=يمين، 2=لا يُرى
      area       : عدد البكسلات الخضراء
      cx_norm    : مركز الجسم الأفقي [0,1]، أو -1 إذا لم يُرَ
    """
    img = camera.getImage()
    if img is None:
        return 2, 0, -1.0

    w = camera.getWidth()
    h = camera.getHeight()

    frame = np.frombuffer(img, np.uint8).reshape((h, w, 4))
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)

    # تنقية الضوضاء بعملية مورفولوجية
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    area = int(cv2.countNonZero(mask))

    if area < 30:
        return 2, 0, -1.0

    M = cv2.moments(mask)
    if M["m00"] == 0:
        return 2, area, -1.0

    cx = int(M["m10"] / M["m00"])
    cx_norm = cx / float(w)

    if cx < w / 3:
        target_dir = -1
    elif cx < 2 * w / 3:
        target_dir = 0
    else:
        target_dir = 1

    return target_dir, area, cx_norm


def move_robot(direction):
    """تحريك الروبوت حسب الاتجاه"""
    speed = 6.0
    if direction == "FORWARD":
        for wh in wheels:
            wh.setVelocity(speed)
    elif direction == "BACKWARD":
        for wh in wheels:
            wh.setVelocity(-speed)
    elif direction == "TURN_RIGHT":
        wheels[0].setVelocity(speed)
        wheels[1].setVelocity(-speed)
        wheels[2].setVelocity(speed)
        wheels[3].setVelocity(-speed)
    elif direction == "TURN_LEFT":
        wheels[0].setVelocity(-speed)
        wheels[1].setVelocity(speed)
        wheels[2].setVelocity(-speed)
        wheels[3].setVelocity(speed)
    elif direction == "STOP":
        for wh in wheels:
            wh.setVelocity(0.0)


def is_arm_stable(target2, target3, tol=0.08):
    """يتحقق من وصول الذراع للموضع المطلوب فعلياً"""
    return (abs(arm_sensors[1].getValue() - target2) < tol and
            abs(arm_sensors[2].getValue() - target3) < tol)


def is_fingers_closed(target=0.008, tol=0.005):
    """يتحقق من إغلاق الأصابع فعلياً"""
    return (left_finger_sensor.getValue() < target + tol and
            right_finger_sensor.getValue() < target + tol)


# ─────────────────────────────────────────────
# 6.  الحلقة الرئيسية
# ─────────────────────────────────────────────
while robot.step(timestep) != -1:

    # ── قراءة الحساسات ──
    pos = gps.getValues()
    dist = ds_front.getValue()
    yaw = imu.getRollPitchYaw()[2]
    t_dir, t_area, t_cx = green_target()

    # ── متجه الحالة (6 عناصر) ──
    state_np = np.array([[
        pos[0],
        pos[2],
        dist / 1000.0,
        yaw,
        float(t_dir),
        min(t_area / 5000.0, 1.0)
    ]], dtype=np.float32)
    state_tensor = torch.from_numpy(state_np)

    # ════════════════════════════════════════
    #  وضع التدريب
    # ════════════════════════════════════════
    if ROBOT_STATE == "TRAINING":

        # اختيار الفعل (ε-greedy)
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(0, 5)
        else:
            with torch.no_grad():
                action_idx = torch.argmax(model(state_tensor)[0]).item()

        move_robot(directions[action_idx])

        # انتظار لرؤية أثر الفعل
        for _ in range(3):
            if robot.step(timestep) == -1:
                break

        # ── الحالة بعد تنفيذ الفعل ──
        new_pos = gps.getValues()
        new_dist = ds_front.getValue()
        new_yaw = imu.getRollPitchYaw()[2]
        new_tdir, new_tarea, new_tcx = green_target()

        if new_tarea > 0:
            last_seen_area = new_tarea

        next_state_np = np.array([[
            new_pos[0],
            new_pos[2],
            new_dist / 1000.0,
            new_yaw,
            float(new_tdir),
            min(new_tarea / 5000.0, 1.0)
        ]], dtype=np.float32)
        next_state_tensor = torch.from_numpy(next_state_np)

        # ─────────────────────────────────────
        # حساب المكافأة المُصلَّحة
        # ─────────────────────────────────────
        goal_reached = False
        target_visible = (new_tdir != 2)

        # ❶ خروج من حدود البيئة
        if abs(new_pos[0]) > ENV_LIMIT_X or abs(new_pos[2]) > ENV_LIMIT_Z:
            reward = -50.0

        # ❷ وصول حقيقي للهدف الأخضر
        #    الشرط المُصلَح: يجب أن يكون الجسم مرئياً + كبيراً + في المنتصف
        #    + حساس المسافة يؤكد وجود جسم قريب أمامه
        elif (target_visible
              and new_tdir == 0
              and new_tarea >= 400
              and new_dist < 300):
            reward = 200.0
            goal_reached = True

        # ❸ اقتراب جيد من الهدف (مرئي + وسط + مساحة متوسطة)
        elif target_visible and new_tdir == 0 and new_tarea >= 150:
            reward = 20.0
            if directions[action_idx] == "FORWARD":
                reward += 10.0

        # ❹ اصطدام بجسم غير الهدف
        elif new_dist < 80 and new_tarea < 50:
            reward = -15.0

        elif new_dist < 150 and new_tarea < 50:
            reward = -5.0

        # ❺ يرى الهدف في المنتصف → تشجيع التقدم
        elif target_visible and new_tdir == 0:
            reward = 8.0
            if directions[action_idx] == "FORWARD":
                reward += 5.0

        # ❻ يرى الهدف في الجانب → تشجيع الدوران الصحيح
        elif target_visible:
            reward = 3.0
            if ((new_tdir == -1 and directions[action_idx] == "TURN_LEFT") or
                    (new_tdir == 1 and directions[action_idx] == "TURN_RIGHT")):
                reward += 5.0
            elif ((new_tdir == -1 and directions[action_idx] == "TURN_RIGHT") or
                  (new_tdir == 1 and directions[action_idx] == "TURN_LEFT")):
                reward -= 2.0

        # ❼ لا يرى الهدف → استكشاف
        else:
            reward = -0.5
            if directions[action_idx] == "STOP":
                reward = -2.0
            if directions[action_idx] in ["TURN_LEFT", "TURN_RIGHT"]:
                reward += 0.5

        # ❽ عقوبة الجمود
        movement = abs(new_pos[0] - pos[0]) + abs(new_pos[2] - pos[2])
        if movement < 0.001 and directions[action_idx] != "STOP":
            reward -= 1.5
            consecutive_stuck += 1
        else:
            consecutive_stuck = 0

        if consecutive_stuck > 10:
            reward -= 3.0
            consecutive_stuck = 0

        # ─────────────────────────────────────
        # تحديث الشبكة
        # ─────────────────────────────────────
        current_q = model(state_tensor)
        with torch.no_grad():
            max_next_q = torch.max(model(next_state_tensor))
            target_q_val = reward + gamma * max_next_q

        target_f = current_q.clone().detach()
        target_f[0][action_idx] = target_q_val

        loss = criterion(current_q, target_f)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # ─────────────────────────────────────
        # الانتقال لمرحلة الإمساك
        # ─────────────────────────────────────
        if goal_reached:
            move_robot("STOP")
            torch.save(model.state_dict(), MODEL_FILE)
            ROBOT_STATE = "PICKING_APPROACH"
            pick_start_time = robot.getTime()
            pick_phase = 0
            print(f"[INFO] وصل للهدف | بدء الإمساك | ε={epsilon:.3f}")

    # ════════════════════════════════════════
    #  مرحلة 1: إنزال الذراع نحو الجسم
    # ════════════════════════════════════════
    elif ROBOT_STATE == "PICKING_APPROACH":
        move_robot("STOP")

        if pick_phase == 0:
            arm_joints[0].setPosition(0.0)
            arm_joints[1].setPosition(PICK_ARM2_DOWN)
            arm_joints[2].setPosition(PICK_ARM3_DOWN)
            arm_joints[3].setPosition(0.0)
            arm_joints[4].setPosition(0.0)
            left_finger.setPosition(PICK_FINGER_OPEN)
            right_finger.setPosition(PICK_FINGER_OPEN)
            pick_phase = 1

        elif pick_phase == 1:
            elapsed = robot.getTime() - pick_start_time
            arm_ok = is_arm_stable(PICK_ARM2_DOWN, PICK_ARM3_DOWN)

            if arm_ok or elapsed > 3.0:
                ROBOT_STATE = "PICKING_CLOSE"
                pick_start_time = robot.getTime()
                pick_phase = 2

    # ════════════════════════════════════════
    #  مرحلة 2: إغلاق الأصابع لإمساك الجسم
    # ════════════════════════════════════════
    elif ROBOT_STATE == "PICKING_CLOSE":
        move_robot("STOP")

        if pick_phase == 2:
            left_finger.setPosition(PICK_FINGER_CLOSED)
            right_finger.setPosition(PICK_FINGER_CLOSED)
            pick_phase = 3

        elif pick_phase == 3:
            elapsed = robot.getTime() - pick_start_time
            fingers_ok = is_fingers_closed(PICK_FINGER_CLOSED)

            if fingers_ok or elapsed > 2.0:
                ROBOT_STATE = "PICKING_LIFT"
                pick_start_time = robot.getTime()
                pick_phase = 4

    # ════════════════════════════════════════
    #  مرحلة 3: رفع الجسم للأعلى
    # ════════════════════════════════════════
    elif ROBOT_STATE == "PICKING_LIFT":
        move_robot("STOP")

        if pick_phase == 4:
            arm_joints[1].setPosition(LIFT_ARM2_UP)
            arm_joints[2].setPosition(LIFT_ARM3_UP)
            # الأصابع تبقى مغلقة
            left_finger.setPosition(PICK_FINGER_CLOSED)
            right_finger.setPosition(PICK_FINGER_CLOSED)
            pick_phase = 5

        elif pick_phase == 5:
            elapsed = robot.getTime() - pick_start_time
            arm_ok = is_arm_stable(LIFT_ARM2_UP, LIFT_ARM3_UP)

            if arm_ok or elapsed > 3.0:
                ROBOT_STATE = "FINISHED"
                torch.save(model.state_dict(), MODEL_FILE)
                print("[INFO] تمّ رفع الجسم بنجاح!")

    # ════════════════════════════════════════
    #  مرحلة الانتهاء
    # ════════════════════════════════════════
    elif ROBOT_STATE == "FINISHED":
        move_robot("STOP")
        break

    # ── حفظ دوري كل 50 ثانية ──
    current_time = int(robot.getTime())
    if current_time > 0 and current_time % 50 == 0:
        torch.save(model.state_dict(), MODEL_FILE)
