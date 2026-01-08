import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.tree import DecisionTreeClassifier

# ===================== CSV 数据与训练模型 =====================
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except:
    base_dir = os.getcwd()

csv_file = os.path.join(base_dir, "door_training_data.csv")

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["speed","body_angle","head_angle","head_alignment","will_enter","timestamp"])
    print("📁 CSV 创建成功 / CSV作成成功")
else:
    print("📁 CSV 已存在 / CSVは既に存在しています")

model = DecisionTreeClassifier(max_depth=5)
model_trained = False

def try_train_model_from_csv():
    """尝试从 CSV 训练模型 / CSVからモデルを訓練してみる"""
    global model, model_trained
    try:
        df = pd.read_csv(csv_file)
        if len(df) < 15:
            print("⚠️ 数据量不够继续收集中… / データ数が不足しています")
            return
        X = df[["speed","body_angle","head_angle","head_alignment"]].values
        y = df["will_enter"].astype(int).values
        model = DecisionTreeClassifier(max_depth=6)
        model.fit(X, y)
        model_trained = True
        print(f"🧠 已训练模型，数据量：{len(df)} / モデルを訓練しました, データ数: {len(df)}")
    except Exception as e:
        print("❌ 模型训练错误 / モデル訓練エラー:", e)

def append_to_csv(batch):
    """批量写入 CSV / CSVにデータをバッチ書き込み"""
    try:
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for row in batch:
                writer.writerow(row + [timestamp])
    except Exception as e:
        print("❌ CSV 写入错误 / CSV書き込みエラー:", e)

# ===================== 未来路径预测函数 =====================
def clamp_angle(angle):
    """角度归一化 [-180,180] / 角度を正規化 [-180,180]"""
    angle = angle % 360
    return angle-360 if angle > 180 else angle

def predict_future_path(px, py, body_angle, head_angle, move_angle, speed,
                     w_move=0.1, w_body=0.8, w_head=0.1, future_steps=40):
    """
    未来路径预测函数 + 意图推定 / 未来経路予測 + 意図推定
    """
    body_rad = np.deg2rad(body_angle)
    head_rad = np.deg2rad(head_angle)
    move_rad = np.deg2rad(move_angle)

    # 三方向加权计算位移 / 3方向の加重速度分解
    dx = speed * (w_body*np.cos(body_rad)+w_head*np.cos(head_rad)+w_move*np.cos(move_rad))
    dy = speed * (w_body*np.sin(body_rad)+w_head*np.sin(head_rad)+w_move*np.sin(move_rad))

    # 未来坐标 / 未来座標生成
    steps = np.arange(1, future_steps+1)
    future_x = px + dx*steps
    future_y = py + dy*steps

    # 是否进入门 / ドアに入るか
    will_enter = np.any((future_x>door_x1)&(future_x<door_x2)&(future_y>door_y1)&(future_y<door_y2))

    # 头部对门方向对齐程度 / 頭部のドア方向整合度
    door_vec = np.array([door_center[0]-px, door_center[1]-py])
    door_vec_norm = door_vec / (np.linalg.norm(door_vec)+1e-6)
    head_vec = np.array([np.cos(head_rad), np.sin(head_rad)])
    head_align = float(np.dot(head_vec, door_vec_norm))

    # 最终开门判断 / 最終ドア開閉判定
    triggered = bool(will_enter and head_align>0.2)

    return triggered, (future_x, future_y), head_align

# ===================== 初始化 YOLO + DeepSORT + Mediapipe =====================
yolo = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
drawer = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
FPS = cap.get(cv2.CAP_PROP_FPS) or 30
PIXEL_TO_METER = 0.005

# ===================== 门参数 =====================
ret, frame = cap.read()
H, W = frame.shape[:2]
center = (W//2, H//2)

door_width, door_height = 800, 640
door_x1, door_x2 = center[0] - door_width//2, center[0] + door_width//2
door_y1, door_y2 = center[1] - door_height//2, center[1] + door_height//2
door_center = (center[0], center[1])

left_pos, right_pos = door_x1, door_x2
door_open = False
max_open = door_width//2
door_speed = 14
hold_time = 1.2
timer = 0

left_pos, right_pos = door_x1, door_x2
door_open = False
max_open = door_width//2
door_speed = 14
hold_time = 1.2
timer = 0

# =====【补丁1】门状态机变量（不删原变量）=====
door_state = "CLOSED"   # CLOSED / OPENING / OPEN / CLOSING
state_timer = 0.0

# trigger 抗抖（持续触发）
last_trigger_time = 0.0
trigger_grace = 0.8     # 秒，允许预测短暂抖动


# ===================== 逐人 Pose 提取函数 =====================
def extract_pose_from_bbox(frame, bbox, pose_model):
    """
    对单个检测框提取 Pose（逐人）
    frame : 原始 BGR 图像
    bbox  : (x1, y1, x2, y2)
    return: pose_landmarks (已映射回原图坐标) or None
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # 防越界
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x2 - x1 < 40 or y2 - y1 < 40:
        return None

    roi = frame[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    result = pose_model.process(roi_rgb)
    if not result.pose_landmarks:
        return None

    # 将 landmark 坐标映射回原图
    for lm in result.pose_landmarks.landmark:
        lm.x = lm.x * (x2 - x1) / w + x1 / w
        lm.y = lm.y * (y2 - y1) / h + y1 / h

    return result.pose_landmarks

# ===================== TRACK历史速度记录 =====================
history = {}
def add_history(id,x,y,max_len=8):
    """记录每个目标的历史中心位置 / 各ターゲットの過去の中心位置を記録"""
    if id not in history: history[id] = []
    history[id].append((x,y))
    if len(history[id])>max_len: history[id].pop(0)

def calc_motion(id):
    """根据历史位置计算移动方向向量和速度 / 履歴位置から移動方向ベクトルと速度を計算"""
    if len(history.get(id,[]))<2: return None,0
    (x1,y1),(x2,y2)=history[id][-2],history[id][-1]
    dx,dy=x2-x1,y2-y1
    speed=np.sqrt(dx*dx+dy*dy)
    return (dx/speed,dy/speed) if speed>1e-5 else None, speed

# ===================== 主循环 =====================
print("🚪 系统启动中，按 ESC 退出 / システム起動中、ESCで終了")
frame_id = 0
trigger = False

while True:
    ret, frame = cap.read()
    if not ret: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections = []
    results = yolo(rgb)[0]

    # YOLO 检测 / YOLO検出
    for box, score, cls in zip(results.boxes.xyxy.cpu().numpy(),
                               results.boxes.conf.cpu().numpy(),
                               results.boxes.cls.cpu().numpy()):
        if int(cls) == 0 and score > 0.3:
            x1, y1, x2, y2 = map(int, box)
            detections.append(([x1, y1, x2-x1, y2-y1], score, 0))

    tracks = tracker.update_tracks(detections, frame=frame)
    trigger = False
    data_batch = []

    for t in tracks:
        if not t.is_confirmed(): continue
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        cx, cy = (x1+x2)//2, (y1+y2)//2

        # 绘制检测框 / バウンディングボックス描画
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID:{t.track_id}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        add_history(t.track_id, cx, cy)
        motion, speed_px = calc_motion(t.track_id)
        speed_ms = speed_px * PIXEL_TO_METER * FPS

        # ================== 逐人 Pose ==================
        pose_landmarks = extract_pose_from_bbox(frame, (x1, y1, x2, y2), pose_model)
        if pose_landmarks is None: continue
        drawer.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = pose_landmarks.landmark

        # 关键点 / キーポイント
        nose = lm[mp_pose.PoseLandmark.NOSE.value]
        left_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
        left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # ------------------- 头部方向 / 頭部方向 -------------------
        head_vec = (nose.x - (left_ear.x + right_ear.x)/2, nose.y - (left_ear.y + right_ear.y)/2)
        hl = np.linalg.norm(head_vec)
        if hl < 1e-5 or motion is None: continue
        head_angle = clamp_angle(np.rad2deg(np.arctan2(head_vec[1], head_vec[0])))

        # ------------------- 身体方向（改进版）/ 体の方向 -------------------
        shoulder_vec = np.array([right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y])
        sl_norm = np.linalg.norm(shoulder_vec)
        if sl_norm < 1e-5: continue
        shoulder_vec_norm = shoulder_vec / sl_norm
        perp_vec1 = np.array([-shoulder_vec_norm[1], shoulder_vec_norm[0]])
        perp_vec2 = np.array([shoulder_vec_norm[1], -shoulder_vec_norm[0]])
        neck = np.array([(left_shoulder.x + right_shoulder.x)/2, (left_shoulder.y + right_shoulder.y)/2])
        nose_vec = np.array([nose.x - neck[0], nose.y - neck[1]])
        if np.dot(perp_vec1, nose_vec) > 0:
            body_vec = perp_vec1
        else:
            body_vec = perp_vec2
        body_angle = clamp_angle(np.rad2deg(np.arctan2(body_vec[1], body_vec[0])))

        body_dir = (int(cx + body_vec[0] * 300), int(cy + body_vec[1] * 300))
        cv2.arrowedLine(frame, (cx, cy), body_dir, (255, 0, 0), 13)
        shoulder_dir = (int(cx + shoulder_vec_norm[0] * 300), int(cy + shoulder_vec_norm[1] * 300))
        cv2.arrowedLine(frame, (cx, cy), shoulder_dir, (0, 0, 255), 5)

        # ------------------- 移动方向 / 移動方向 -------------------
        move_angle = np.rad2deg(np.arctan2(motion[1], motion[0]))
        move_end = (cx + int(motion[0]*60), cy + int(motion[1]*60))
        cv2.arrowedLine(frame, (cx,cy), move_end, (200,150,255), 13)

        # ------------------- 绘制头部箭头 -------------------
        nose_px = (int(nose.x*W), int(nose.y*H))
        head_dir = (nose_px[0]+int(head_vec[0]*3000), nose_px[1]+int(head_vec[1]*3000))
        cv2.arrowedLine(frame, nose_px, head_dir, (0,255,0), 3)

        # ------------------- 未来路径预测 / 未来経路予測 -------------------
        triggered, future_path, head_align = predict_future_path(cx, cy, body_angle, head_angle, move_angle, speed_ms)
        if triggered: 
            trigger = True
            last_trigger_time = time.time()   # ←【补丁2】记录最近触发时间

        fx, fy = future_path
        for i in range(1, len(fx), 2):
            cv2.line(frame, (int(fx[i-1]), int(fy[i-1])), (int(fx[i]), int(fy[i])), (0,255,255), 12)

        data_batch.append([speed_ms, body_angle, head_angle, round(head_align,4), int(triggered)])

    # =====【补丁3】把瞬时 trigger 变成稳定 trigger =====
    trigger_persistent = (time.time() - last_trigger_time) <= trigger_grace

    append_to_csv(data_batch)
    if frame_id % 300 == 0:
        try_train_model_from_csv()

    # ------------------- 门动画逻辑 / ドアアニメーション -------------------
    """
    if trigger:
        door_open = True
        timer = time.time()
    if door_open:
        left_pos = max(left_pos - door_speed, door_x1 - max_open)
        right_pos = min(right_pos + door_speed, door_x2 + max_open)
        if time.time() - timer > hold_time and not trigger:
            door_open = False
    else:
        left_pos = min(left_pos + door_speed, door_x1)
        right_pos = max(right_pos - door_speed, door_x2)
    """

    # =====【补丁4】门状态机逻辑（只控制门，不碰其他）=====
    if door_state == "CLOSED":
        if trigger_persistent:
            door_state = "OPENING"

    elif door_state == "OPENING":
        left_pos = max(left_pos - door_speed, door_x1 - max_open)
        right_pos = min(right_pos + door_speed, door_x2 + max_open)

        # 门真正开到位
        if left_pos <= door_x1 - max_open and right_pos >= door_x2 + max_open:
            door_state = "OPEN"
            state_timer = time.time()

    elif door_state == "OPEN":
        # 有人就续命
        if trigger_persistent:
            state_timer = time.time()
        elif time.time() - state_timer > hold_time:
            door_state = "CLOSING"

    elif door_state == "CLOSING":
        left_pos = min(left_pos + door_speed, door_x1)
        right_pos = max(right_pos - door_speed, door_x2)

        if left_pos >= door_x1 and right_pos <= door_x2:
            door_state = "CLOSED"

   # 最后绘制门边框 + 半透明门板
    overlay = frame.copy()
    cv2.rectangle(overlay, (left_pos,door_y1), (left_pos+door_width//2,door_y2), (0,0,255), -1)
    cv2.rectangle(overlay, (right_pos-door_width//2,door_y1), (right_pos,door_y2), (0,0,255), -1)

    alpha = 0.4  # 透明度，0~1，可调节
    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

    # 绘制门边框
    cv2.rectangle(frame, (door_x1,door_y1), (door_x2,door_y2), (0,0,0), 10)

    cv2.putText(frame, f"Door: {'OPEN' if door_open else 'CLOSED'}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    cv2.imshow("Smart Auto Door AI", frame)
    if cv2.waitKey(1) & 0xFF == 27: break
    frame_id += 1

cap.release()
cv2.destroyAllWindows()
print("📍 结束运行，数据已写入:", csv_file)
