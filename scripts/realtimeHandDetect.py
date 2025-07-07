import cv2
import mediapipe as mp

# MediaPipeの初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# カメラキャプチャ開始
cap = cv2.VideoCapture(0)

def is_finger_up(hand_landmarks, finger_indices):
    return hand_landmarks.landmark[finger_indices[0]].y < hand_landmarks.landmark[finger_indices[1]].y

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR→RGB変換
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 手の検出
    results = hands.process(rgb)

    gesture = "なし"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 指の状態を判定（とてもシンプルに）
            finger_up = []
            finger_up.append(is_finger_up(hand_landmarks, [8, 6]))   # 人差し指
            finger_up.append(is_finger_up(hand_landmarks, [12, 10])) # 中指
            finger_up.append(is_finger_up(hand_landmarks, [16, 14])) # 薬指
            finger_up.append(is_finger_up(hand_landmarks, [20, 18])) # 小指

            if all(finger_up):
                gesture = "パー"
            elif not any(finger_up):
                gesture = "グー"
            else:
                gesture = "その他"

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ジェスチャー名を表示
    cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ウィンドウ表示
    cv2.imshow("Hand Gesture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
