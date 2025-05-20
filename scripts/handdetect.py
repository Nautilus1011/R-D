import cv2
import mediapipe as mp
import os

# MediaPipeの準備
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 入力画像フォルダと出力フォルダのパス
input_dir = "/home/ruitoby/workspace/test/images/hands"
output_dir = "/home/ruitoby/workspace/test/runs/detect"
os.makedirs(output_dir, exist_ok=True)

# Handsモデルの初期化（静止画モード）
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7
) as hands:

    # 入力フォルダ内のすべての画像ファイルを処理
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"{filename} の読み込みに失敗しました。スキップします。")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                print(f"{filename} に手が検出されませんでした。")

            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)
            print(f"保存完了: {output_path}")
