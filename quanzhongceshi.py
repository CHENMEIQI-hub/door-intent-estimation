import csv
import numpy as np
import random
import math

# ============================================
# パラメータ設定
# ============================================

REPEAT_RUNS = 10          # 各重みセットの繰り返し回数
TOTAL_FRAMES = 100000     # 各シミュレーションの総フレーム数

DOOR_X = 500
DOOR_Y_MIN = 350
DOOR_Y_MAX = 450

LEFT_DOOR_X_MIN = 480
LEFT_DOOR_X_MAX = 500
RIGHT_DOOR_X_MIN = 500
RIGHT_DOOR_X_MAX = 520

# ============================================
# 重み組み合わせの生成（36 種類）
# ============================================

weight_sets = []
ws = np.arange(0.1, 0.81, 0.1)

for w_move in ws:
    for w_body in ws:
        for w_head in ws:
            if abs((w_move + w_body + w_head) - 1.0) < 1e-6:
                weight_sets.append((round(w_move, 2), round(w_body, 2), round(w_head, 2)))

print("重み組み合わせ数 =", len(weight_sets))  # 36 のはず


# ============================================
# ドアパネル衝突判定
# ============================================

def hit_left_door_panel(x, y):
    return LEFT_DOOR_X_MIN <= x <= LEFT_DOOR_X_MAX and DOOR_Y_MIN <= y <= DOOR_Y_MAX

def hit_right_door_panel(x, y):
    return RIGHT_DOOR_X_MIN <= x <= RIGHT_DOOR_X_MAX and DOOR_Y_MIN <= y <= DOOR_Y_MAX


# ============================================
# 将来経路予測
# ============================================

def predict_future_path(px, py, body_angle, head_angle, move_angle, speed,
                        w_move, w_body, w_head, steps=40):

    blended_angle = w_move * move_angle + w_body * body_angle + w_head * head_angle
    rad = math.radians(blended_angle)

    fx = px + math.cos(rad) * speed * steps
    fy = py + math.sin(rad) * speed * steps

    toward_door = fx > DOOR_X - 30 and DOOR_Y_MIN <= fy <= DOOR_Y_MAX
    return toward_door


# ============================================
# ランダムな人物生成
# ============================================

def new_person():
    return {
        "x": random.randint(100, 400),
        "y": random.randint(200, 600),
        "body": random.randint(0, 360),
        "head": random.randint(0, 360),
        "move": random.randint(0, 360),
        "speed": random.uniform(1.0, 2.0)
    }


# ============================================
# 単回シミュレーション
# ============================================

def run_simulation(w_move, w_body, w_head):

    collision_frames = 0
    false_open_frames = 0
    correct_frames = 0

    p = new_person()
    door_open = 0.0

    for frame in range(TOTAL_FRAMES):

        # 姿勢のランダム揺らぎ
        p["move"] = (p["move"] + random.uniform(-8, 8)) % 360
        p["body"] = (p["body"] + random.uniform(-5, 5)) % 360
        p["head"] = (p["head"] + random.uniform(-10, 10)) % 360

        # 進行方向予測
        toward_door = predict_future_path(
            p["x"], p["y"], p["body"], p["head"], p["move"], p["speed"],

            w_move, w_body, w_head
        )

        # ドア制御
        if toward_door:
            door_open = 1.0
        else:
            door_open = max(0.0, door_open - 0.05)

        # 移動処理
        rad = math.radians(p["move"])
        p["x"] += math.cos(rad) * p["speed"]
        p["y"] += math.sin(rad) * p["speed"]

        # ドア衝突
        if hit_left_door_panel(p["x"], p["y"]) or hit_right_door_panel(p["x"], p["y"]):
            collision_frames += 1

        # 誤開扉
        if door_open > 0.3 and not toward_door:
            false_open_frames += 1

        # 正判定
        if (toward_door and door_open > 0.3) or (not toward_door and door_open <= 0.3):
            correct_frames += 1

        # 範囲外 → リスポーン
        if p["x"] < 0 or p["x"] > 900 or p["y"] < 0 or p["y"] > 800:
            p = new_person()

    total_error = collision_frames + false_open_frames
    error_rate = (total_error / TOTAL_FRAMES) * 100
    correct_rate = (correct_frames / TOTAL_FRAMES) * 100

    return collision_frames, false_open_frames, total_error, error_rate, correct_rate


# ============================================
# CSV 出力（平均値のみ）
# ============================================

with open("weight_test_result_avg.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "w_move", "w_body", "w_head",
        "avg_collision_frames",
        "avg_false_open_frames",
        "avg_total_error_frames",
        "avg_error_rate(%)",
        "avg_correct_rate(%)"
    ])


best_result = None
all_results = []   # (w_move, w_body, w_head, avg_total_error, avg_correct_rate)


# ============================================
# メインループ：各重みセットを複数回実行して平均化
# ============================================

for w_move, w_body, w_head in weight_sets:

    sum_collision = 0
    sum_false_open = 0
    sum_total_error = 0
    sum_error_rate = 0.0
    sum_correct_rate = 0.0

    print(f"\n>>> 重み {w_move}, {w_body}, {w_head} を実行中…")

    for i in range(REPEAT_RUNS):
        c, f, te, er, cr = run_simulation(w_move, w_body, w_head)

        sum_collision += c
        sum_false_open += f
        sum_total_error += te
        sum_error_rate += er
        sum_correct_rate += cr

        print(f"  {i+1}/{REPEAT_RUNS} 回目完了… エラーフレーム={te}")

    # 平均化
    avg_collision = sum_collision / REPEAT_RUNS
    avg_false_open = sum_false_open / REPEAT_RUNS
    avg_total_error = sum_total_error / REPEAT_RUNS
    avg_error_rate = sum_error_rate / REPEAT_RUNS
    avg_correct_rate = sum_correct_rate / REPEAT_RUNS

    # 集計用に保存
    all_results.append((w_move, w_body, w_head, avg_total_error, avg_correct_rate))

    # 最適重み判定（エラー最小）
    if best_result is None or avg_total_error < best_result["avg_total_error"]:
        best_result = {
            "weights": (w_move, w_body, w_head),
            "avg_total_error": avg_total_error,
            "avg_error_rate": avg_error_rate
        }

    # CSV 書き込み
    with open("weight_test_result_avg.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            w_move, w_body, w_head,
            avg_collision, avg_false_open, avg_total_error,
            f"{avg_error_rate:.2f}%",
            f"{avg_correct_rate:.2f}%"
        ])


# ============================================
# 全重みの平均結果（集計表示）
# ============================================

print("\n=======================================")
print("📊 全重みの平均エラーフレーム＆平均正解率（集計）")
print("=======================================")

for w_move, w_body, w_head, avg_error, avg_correct in all_results:
    print(f"重み({w_move}, {w_body}, {w_head}) → 平均エラーフレーム={avg_error:.2f}, 平均正解率={avg_correct:.2f}%")


# ============================================
# 正解率 TOP 5
# ============================================

print("\n=======================================")
print("🏆 正解率トップ5の重みとエラーフレーム")
print("=======================================")

top5 = sorted(all_results, key=lambda x: x[4], reverse=True)[:5]

for rank, (w_move, w_body, w_head, avg_error, avg_correct) in enumerate(top5, start=1):
    print(f"第 {rank} 位 → 重み({w_move}, {w_body}, {w_head})：正解率={avg_correct:.2f}%，平均エラー={avg_error:.2f}")


# ============================================
# 最適重み（エラー最小）
# ============================================

print("\n================================")
print("⭐ エラー最少の最適重み ⭐")
print("================================")
w_move, w_body, w_head = best_result["weights"]
print(f"最適重み → w_move={w_move}, w_body={w_body}, w_head={w_head}")
print("最小平均エラーフレーム =", best_result["avg_total_error"])
print(f"平均エラー率 = {best_result['avg_error_rate']:.2f}%")
print(f"平均正解率 = {100 - best_result['avg_error_rate']:.2f}%")
print("================================")
print("\n完了しました！結果は weight_test_result_avg.csv に保存されています。")
