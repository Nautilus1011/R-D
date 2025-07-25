import cv2
import numpy as np
import onnxruntime
import time # 処理速度確認用にtimeモジュールを追加

# ------------ NMS関連の関数定義 ------------
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def non_max_suppression(detections, iou_threshold=0.5):
    # 信頼度が高い順にソート
    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    results = []
    while detections:
        best = detections.pop(0) # 最も信頼度の高い検出を取得
        results.append(best)
        # 取得した検出とIoUが高いものを除去
        detections = [d for d in detections if iou(best[0], d[0]) < iou_threshold]
    return results


# ------------ モデルとラベル読み込み ------------
# モデルとラベルファイルのパスは環境に合わせて変更してください
model_path = "../../models/onnx_weights/yolov5s.onnx"
names_path = "../../models/coco.names"

try:
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
except Exception as e:
    print(f"ONNXモデルの読み込みに失敗しました: {e}")
    print(f"パス: {model_path} が正しいか確認してください。")
    exit()

try:
    with open(names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"クラス名ファイル '{names_path}' が見つかりません。パスが正しいか確認してください。")
    exit()
except Exception as e:
    print(f"クラス名ファイルの読み込みに失敗しました: {e}")
    exit()


# ------------ カメラ起動 ------------
cap = cv2.VideoCapture(0) # 0は通常、デフォルトのウェブカメラ
if not cap.isOpened():
    print("カメラが開けませんでした。カメラが接続されているか、他のアプリケーションで使用されていないか確認してください。")
    exit()

print("リアルタイム物体検出を開始します。'Ctrl+C' で終了します。")
print("-" * 50)

# ------------ メインループ ------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームの読み込みに失敗しました。")
            break

        start_time = time.time() # 処理時間計測開始

        # 前処理: YOLOv5sモデルの標準的な入力に合わせて調整
        # (BGR -> RGB, リサイズ 640x640, 転置 HWC -> CHW, 正規化 0-1, バッチ次元追加)
        resized = cv2.resize(frame, (640, 640))
        rgb_frame = resized[:, :, ::-1] # BGRからRGBへ変換
        input_img = rgb_frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_img, axis=0)

        # 推論
        outputs = session.run([output_name], {input_name: input_tensor})[0]

        # 元画像のサイズを取得 (後処理でボックスを元のスケールに戻すために使用)
        original_h, original_w = frame.shape[:2]

        # 候補抽出とスコア計算
        candidates = []
        # outputs[0] は (1, num_detections, 5 + num_classes) の形状を持つ
        # したがって、ループは outputs[0][0] に対して実行する
        for det in outputs[0]:
            objectness_conf = det[4] # オブジェクトネススコア (物体が存在する信頼度)
            class_scores = det[5:]   # 各クラスのスコア

            # 最も高いクラススコアとそのインデックスを取得
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id] # そのクラスの信頼度

            # 最終的な信頼度 = オブジェクトネススコア * クラススコア
            final_conf = objectness_conf * class_conf

            # 信頼度閾値でフィルタリング
            if final_conf > 0.4: # ここは調整可能です (例: 0.2, 0.5など)
                x_center, y_center, width, height = det[0], det[1], det[2], det[3]

                # バウンディングボックス座標を元の画像サイズにスケール戻し
                # ONNXモデルの出力が640x640スケールに基づいているため、元の画像サイズに合わせて調整
                x1 = int((x_center - width / 2) * (original_w / 640))
                y1 = int((y_center - height / 2) * (original_h / 640))
                x2 = int((x_center + width / 2) * (original_w / 640))
                y2 = int((y_center + height / 2) * (original_h / 640))

                # NMSのためにボックス座標を整数型に変換
                box = (x1, y1, x2, y2)
                candidates.append((box, final_conf, class_id))

        # NMS (Non-Maximum Suppression) で重複除去
        results = non_max_suppression(candidates, iou_threshold=0.5) # IoU閾値も調整可能

        end_time = time.time() # 処理時間計測終了
        process_time = end_time - start_time

        # 結果をコンソールに表示
        detected_objects_info = []
        for box, conf, class_id in results:
            object_name = class_names[class_id] if class_id < len(class_names) else f"Unknown({class_id})"
            detected_objects_info.append(f"{object_name} (信頼度: {conf:.2f})")

        if detected_objects_info:
            print(f"検出された物体 ({process_time:.3f}秒): " + ", ".join(detected_objects_info))
        else:
            print(f"検出された物体はありません ({process_time:.3f}秒)。")

        # 非常に高速なループになるため、適度な遅延を入れるとCPU負荷が下がる場合があります
        # cv2.waitKey(1) は通常画面表示とセットで使うが、ループ制御に使うことも可能。
        # 今回は画面表示しないため不要だが、必要なら time.sleep(0.01) などで調整。

except KeyboardInterrupt:
    print("\n'Ctrl+C' が押されました。アプリケーションを終了します。")

finally:
    cap.release()
    print("カメラリソースを解放しました。")