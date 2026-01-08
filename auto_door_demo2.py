import cv2
import numpy as np
import os
import csv
import pandas as pd
import time
from datetime import datetime


# ===================== 参数设置 =====================
WIDTH, HEIGHT = 900, 600
FPS = 30
NUM_PEOPLE = 4

# 小人半径（用于碰撞判定）
PERSON_RADIUS = 10

door_center = (WIDTH // 2, HEIGHT // 2 + 50)
door_width = 280
door_height = 150
door_open_level = 0.0  # 0~1，表示开门程度


speed_list = [1, 2, 3, 4, 5]
body_angle_list = list(range(-180, 181, 15))
head_angle_offset = list(range(-90, 91, 15))
mode_change_interval = 150


# ===================== 数据文件 =====================
csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "door_training_data.csv")
try:
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["speed", "body_angle", "head_angle", "head_alignment", "will_enter", "timestamp"])
        print("✅ CSV 文件已创建，包含时间戳表头")
    else:
        print("⚠ CSV 文件已存在")
except Exception as e:
    print("❌ CSV 创建失败:", e)


# ===================== 初始化人物 =====================
people = []
for _ in range(NUM_PEOPLE):
    person = {
        "x": int(door_center[0] + np.random.randint(-200, 200)),
        "y": int(door_center[1] + np.random.randint(door_height // 2 + 40, door_height // 2 + 160)),
        
        "prev_x": None,
        "prev_y": None,
        
        "speed": int(np.random.choice(speed_list)),
        "body": int(np.random.choice(body_angle_list)),
        "head": 0,
        "color": tuple(np.random.randint(100, 255, 3).tolist())
    }
    person["head"] = int(person["body"] + np.random.choice(head_angle_offset))
    if person["head"] > 180:
        person["head"] -= 360
    if person["head"] < -180:
        person["head"] += 360
    people.append(person)
    person["wall_following"] = False



print("按 ESC 退出程序")


# ===================== 写入 CSV 函数 =====================
def append_to_csv(data_batch):
    try:
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for row in data_batch:
                writer.writerow(row + [timestamp])
    except Exception as e:
        print("❌ 写入 CSV 失败：", e)


# ===================== 辅助函数 =====================
def clamp_angle(angle):
    a = angle % 360
    if a > 180:
        a -= 360
    return int(a)


def draw_door(frame, open_level):
    wall_thickness = 10
    y_top = door_center[1] - wall_thickness // 2
    y_bottom = door_center[1] + wall_thickness // 2

    x_left = door_center[0] - door_width // 2
    x_right = door_center[0] + door_width // 2

    cv2.rectangle(frame, (x_left - 300, y_top), (x_left, y_bottom), (150, 150, 150), -1)
    cv2.rectangle(frame, (x_right, y_top), (x_right + 300, y_bottom), (150, 150, 150), -1)

    open_len = int((door_width // 2) * open_level)

    cv2.rectangle(frame, (x_left, y_top), (x_right, y_bottom), (180, 255, 180), -1)

    cv2.rectangle(frame, (x_left, y_top),
                  (x_left + (door_width // 2 - open_len), y_bottom), (0, 0, 255), -1)
    cv2.rectangle(frame,
                  (x_right - (door_width // 2 - open_len), y_top),
                  (x_right, y_bottom), (0, 0, 255), -1)


def person_in_door_area(px, py):
    door_x1 = door_center[0] - door_width // 2
    door_x2 = door_center[0] + door_width // 2
    door_y1 = door_center[1] - door_height // 2
    door_y2 = door_center[1] + door_height // 2

    return (
        (px > door_x1 - PERSON_RADIUS) and
        (px < door_x2 + PERSON_RADIUS) and
        (py > door_y1 - PERSON_RADIUS) and
        (py < door_y2 + PERSON_RADIUS)
    )


# ===================== 三向量综合预测 =====================
def predict_future_path_combined(
    px, py,
    body_vec, head_vec, move_vec,
    speed,
    w_move=0.1, w_body=0.8, w_head=0.1,
    future_steps=40
):
    """
    px, py        : 当前真实位置
    body_vec      : 身体朝向单位向量
    head_vec      : 头部朝向单位向量
    move_vec      : 真实移动方向（由上一帧位移得到）
    speed         : 当前速度
    """

    # ===================== 预测用：墙 / 门板阻挡判断 =====================
    wall_thickness = 10

    door_x1 = door_center[0] - door_width // 2
    door_x2 = door_center[0] + door_width // 2
    door_y1 = door_center[1] - wall_thickness // 2
    door_y2 = door_center[1] + wall_thickness // 2

    left_wall_x1 = door_x1 - 300
    left_wall_x2 = door_x1
    right_wall_x1 = door_x2
    right_wall_x2 = door_x2 + 300

    def is_blocked(x, y):
        if (door_y1 - PERSON_RADIUS) <= y <= (door_y2 + PERSON_RADIUS):
            if (left_wall_x1 - PERSON_RADIUS) <= x <= (left_wall_x2 + PERSON_RADIUS):
                return True
            if (right_wall_x1 - PERSON_RADIUS) <= x <= (right_wall_x2 + PERSON_RADIUS):
                return True
        return False


    # ===================== 合成预测速度向量 =====================
    v = speed * (
        w_body * body_vec +
        w_head * head_vec +
        w_move * move_vec
    )

    dx, dy = v

    # ===================== 未来轨迹 =====================
    future_x = []
    future_y = []

    cx, cy = px, py

    for _ in range(future_steps):
        cx += dx
        cy += dy

        # 🚫 预测路径撞墙 → 立刻停止
        if is_blocked(cx, cy):
            break

        future_x.append(cx)
        future_y.append(cy)

    future_x = np.array(future_x)
    future_y = np.array(future_y)


    # ===================== 门区域 =====================
    door_x1 = door_center[0] - door_width // 2
    door_x2 = door_center[0] + door_width // 2
    door_y1 = door_center[1] - door_height // 2
    door_y2 = door_center[1] + door_height // 2

    # 是否会进入门区域（考虑人半径）
    will_enter = np.any(
        (future_x > door_x1 ) &
        (future_x < door_x2 ) &
        (future_y > door_y1 ) &
        (future_y < door_y2 )
    )
    """
    will_enter = np.any(
        (future_x > (door_x1 - PERSON_RADIUS)) &
        (future_x < (door_x2 + PERSON_RADIUS)) &
        (future_y > (door_y1 - PERSON_RADIUS)) &
        (future_y < (door_y2 + PERSON_RADIUS))
    )
    """

    # ===================== 头部是否对准门 =====================
    door_vec = np.array([door_center[0] - px, door_center[1] - py])
    norm = np.linalg.norm(door_vec) + 1e-6
    door_dir = door_vec / norm

    head_alignment = float(np.dot(door_dir, head_vec))

    # ===================== 最终触发条件 =====================
    triggered = (will_enter and head_alignment > -0.1) #角度95.7度

    return triggered, (future_x, future_y), head_alignment



# ===================== 门状态机 =====================
DOOR_CLOSED = 0
DOOR_OPENING = 1
DOOR_OPEN = 2
DOOR_CLOSING = 3

door_state = DOOR_CLOSED
open_speed = 0.1
close_speed = 0.05
no_trigger_frames = 0
close_delay_frames = 30


# ===================== 主循环 =====================
sim_idx = 0

while True:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    if sim_idx % mode_change_interval == 0:
        for p in people:
            p["speed"] = int(np.random.choice(speed_list))
            p["body"] = int(np.random.choice(body_angle_list))
            p["head"] = clamp_angle(p["body"] + int(np.random.choice(head_angle_offset)))
            p["x"] = int(door_center[0] + np.random.randint(-200, 200))
            p["y"] = int(door_center[1] + np.random.randint(door_height // 2 + 20,
                                                           door_height // 2 + 160))

    any_trigger = False
    data_batch = []

    draw_door(frame, door_open_level)

    for p in people:
        p["prev_x"] = p["x"]
        p["prev_y"] = p["y"]
        
        dx = p["speed"] * np.cos(np.deg2rad(p["body"]))
        dy = p["speed"] * np.sin(np.deg2rad(p["body"]))
        new_x = p["x"] + dx
        new_y = p["y"] + dy

        wall_thickness = 10
        door_x1 = door_center[0] - door_width // 2
        door_x2 = door_center[0] + door_width // 2
        door_y1 = door_center[1] - wall_thickness // 2
        door_y2 = door_center[1] + wall_thickness // 2

        left_wall_x1 = door_x1 - 300
        left_wall_x2 = door_x1
        right_wall_x1 = door_x2
        right_wall_x2 = door_x2 + 300

        open_len = int((door_width // 2) * door_open_level)
        door_left_x1 = door_x1
        door_left_x2 = door_x1 + (door_width // 2 - open_len)
        door_right_x1 = door_x2 - (door_width // 2 - open_len)
        door_right_x2 = door_x2

        def hit_left_door_panel(x, y):
            # 门板区域向外扩张 PERSON_RADIUS
            return (
                (door_y1 - PERSON_RADIUS) <= y <= (door_y2 + PERSON_RADIUS) and
                (door_left_x1 - PERSON_RADIUS) <= x <= (door_left_x2 + PERSON_RADIUS)
            )

        def hit_right_door_panel(x, y):
              # 门板区域向外扩张 PERSON_RADIUS
            return (
                (door_y1 - PERSON_RADIUS) <= y <= (door_y2 + PERSON_RADIUS) and
                (door_right_x1 - PERSON_RADIUS) <= x <= (door_right_x2 + PERSON_RADIUS)
            )
        
        def is_in_wall(x, y):
            # 整体墙区域在 y 方向扩张 PERSON_RADIUS（避免穿墙），
            # # 在 x 方向分别扩张左右墙体
            if (door_y1 - PERSON_RADIUS) <= y <= (door_y2 + PERSON_RADIUS):
                if (left_wall_x1 - PERSON_RADIUS) <= x <= (left_wall_x2 + PERSON_RADIUS):
                    return True
                if (right_wall_x1 - PERSON_RADIUS) <= x <= (right_wall_x2 + PERSON_RADIUS):
                    return True
                return False


        if hit_left_door_panel(new_x, new_y):
            p["body"] = clamp_angle(-75)
            p["head"] = clamp_angle(p["body"] + np.random.choice(head_angle_offset))
            p["x"] += p["speed"] * 0.3
            p["y"] -= p["speed"] * 0.1

        elif hit_right_door_panel(new_x, new_y):
            p["body"] = clamp_angle(-105)
            p["head"] = clamp_angle(p["body"] + np.random.choice(head_angle_offset))
            p["x"] -= p["speed"] * 0.3
            p["y"] -= p["speed"] * 0.1

        # ⭐【修改点】开始：撞墙 → 转向搜索（不再贴墙）
        elif is_in_wall(new_x, new_y):
            p["wall_following"] = True
            turned = False
            for delta in [-60, -30, 30, 60]:
                test_body = clamp_angle(p["body"] + delta)
                tx = p["x"] + p["speed"] * np.cos(np.deg2rad(test_body))
                ty = p["y"] + p["speed"] * np.sin(np.deg2rad(test_body))
                if not is_in_wall(tx, ty):
                    p["body"] = test_body
                    p["head"] = clamp_angle(p["body"] + np.random.choice(head_angle_offset))
                    p["x"] = tx
                    p["y"] = ty
                    turned = True
                    break
            if not turned:
                p["body"] = clamp_angle(p["body"] + 180)

        else:
            p["wall_following"] = False   # ⭐ 清除贴墙状态
            p["x"] = new_x
            p["y"] = new_y

        margin = PERSON_RADIUS + 40
        clip_margin = PERSON_RADIUS + 50
        if p["x"] < margin or p["x"] > WIDTH - margin or p["y"] < margin or p["y"] > HEIGHT - margin:
            p["body"] = clamp_angle(p["body"] + 180)
            p["head"] = clamp_angle(p["body"] + np.random.choice(head_angle_offset))
            p["x"] = np.clip(p["x"], clip_margin, WIDTH - clip_margin)
            p["y"] = np.clip(p["y"], clip_margin, HEIGHT - clip_margin)
        
        # ===================== 第 3 步：真实移动方向（move_vec）=====================
        real_dx = p["x"] - p["prev_x"]
        real_dy = p["y"] - p["prev_y"]
        
        norm = np.hypot(real_dx, real_dy)
        if norm > 1e-6:
            move_vec = np.array([real_dx / norm, real_dy / norm])
        else:
            move_vec = np.array([0.0, 0.0])
        
        body_vec = np.array([
            np.cos(np.deg2rad(p["body"])),
            np.sin(np.deg2rad(p["body"]))
        ])
        
        head_vec = np.array([
            np.cos(np.deg2rad(p["head"])),
            np.sin(np.deg2rad(p["head"]))
        ])


        if p["wall_following"]:
            # 🚫 贴墙时：不信身体、不信头，只信真实移动方向
            triggered, (fx, fy), align = predict_future_path_combined(
                p["x"], p["y"],
                body_vec, head_vec, move_vec,
                speed=p["speed"],
                w_body=0.0,
                w_head=0.0,
                w_move=1.0
            )
        else:
            # ✅ 正常情况：三向量综合预测
            triggered, (fx, fy), align = predict_future_path_combined(
                p["x"], p["y"],
                body_vec, head_vec, move_vec,
                speed=p["speed"]
            )


        in_door = person_in_door_area(p["x"], p["y"])
        
        if triggered or in_door:
            any_trigger = True
            no_trigger_frames = 0

        for i in range(1, len(fx), 2):
            cv2.line(frame,
                     (int(fx[i - 1]), int(fy[i - 1])),
                     (int(fx[i]), int(fy[i])),
                     (0, 255, 255), 1)

        cv2.circle(frame, (int(p["x"]), int(p["y"])), PERSON_RADIUS, p["color"], -1)

        move_tip = (int(p["x"] + 30 * move_vec[0]),
                    int(p["y"] + 30 * move_vec[1]))
        body_tip = (int(p["x"] + 40 * np.cos(np.deg2rad(p["body"]))),
                    int(p["y"] + 40 * np.sin(np.deg2rad(p["body"]))))
        head_tip = (int(p["x"] + 30 * np.cos(np.deg2rad(p["head"]))),
                    int(p["y"] + 30 * np.sin(np.deg2rad(p["head"]))))

        cv2.arrowedLine(frame, (int(p["x"]), int(p["y"])), move_tip, (200, 150, 255), 2, tipLength=0.3)  # 粉色：移动方向
        cv2.arrowedLine(frame, (int(p["x"]), int(p["y"])), body_tip, (255, 0, 0), 2, tipLength=0.3)  # 蓝色：身体
        cv2.arrowedLine(frame, (int(p["x"]), int(p["y"])), head_tip, (0, 255, 0), 2, tipLength=0.3)   # 绿色：头部

        color_rule = (0, 255, 0) if (triggered and not p["wall_following"]) else (0, 0, 255)

        cv2.circle(frame, (int(p["x"]), int(p["y"])), PERSON_RADIUS + 4, color_rule, 2)

        data_batch.append([int(p["speed"]), int(p["body"]),
                           int(p["head"]), round(align, 4), int(triggered)])

    append_to_csv(data_batch)


    if door_state == DOOR_CLOSED:
        if any_trigger:
            door_state = DOOR_OPENING
    elif door_state == DOOR_OPENING:
        door_open_level += open_speed
        if door_open_level >= 1.0:
            door_open_level = 1.0
            door_state = DOOR_OPEN
    elif door_state == DOOR_OPEN:
        if not any_trigger:
            no_trigger_frames += 1
            if no_trigger_frames >= close_delay_frames:
                door_state = DOOR_CLOSING
        else:
            no_trigger_frames = 0
    elif door_state == DOOR_CLOSING:
        door_open_level -= close_speed
        if door_open_level <= 0.0:
            door_open_level = 0.0
            door_state = DOOR_CLOSED
            no_trigger_frames = 0

    #draw_door(frame, door_open_level)

    state_name = {0: "CLOSED", 1: "OPENING", 2: "OPEN", 3: "CLOSING"}[door_state]
    cv2.putText(frame, f"Door {state_name}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"Sim Frame: {sim_idx}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    cv2.imshow("Smart Auto Door Simulation (Top-down Thin Door)", frame)
    sim_idx += 1

    key = cv2.waitKey(int(1000 / FPS)) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
print(f"✅ 模拟结束，数据已保存至 {csv_file}")
