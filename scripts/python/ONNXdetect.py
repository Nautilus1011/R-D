import cv2
import numpy as np
import onnxruntime as ort
import time

# --- 設定 ---
# ONNXモデルのパス
ONNX_MODEL_PATH = '../../models/onnx_weights/yolov5s.onnx' # あなたのONNXモデルのファイル名に置き換えてください
# クラスラベルファイルのパス
CLASSES_FILE = '../../models/coco.names' # あなたのラベルファイル名に置き換えてください

# モデルの入力サイズ (例: YOLOv5sは640x640)
# モデルによって異なるので、使用するモデルの入力サイズに合わせてください
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# 信頼度（confidence）の閾値: この値より低い検出は表示しない
CONF_THRESHOLD = 0.4
# 非最大抑制（NMS）の閾値: 重複するバウンディングボックスを除去する
NMS_THRESHOLD = 0.5

# --- クラスラベルの読み込み ---
def load_classes(file_path):
    with open(file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

classes = load_classes(CLASSES_FILE)

# --- ONNX Runtimeセッションの初期化 ---
# 利用可能なプロバイダを確認し、CPUExecutionProviderを使用
# もしCUDAなどが利用可能であれば、'CUDAExecutionProvider'などを指定することも可能
# provider = ['CPUExecutionProvider']
# Raspberry Piであれば、通常はCPUExecutionProviderで十分です
provider = [
    'CPUExecutionProvider'
]

try:
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=provider)
    print(f"ONNX Runtime Session initialized successfully with providers: {session.get_providers()}")
except Exception as e:
    print(f"ONNX Runtime Session initialization failed: {e}")
    print("Please ensure your ONNX model path is correct and onnxruntime is properly installed.")
    exit()

# モデルの入力・出力情報の取得
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]
print(f"Model Input Name: {input_name}")
print(f"Model Output Names: {output_names}")


# --- 前処理関数 ---
def preprocess(image):
    # 画像のリサイズ
    input_image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    # 画像の正規化 (0-255 -> 0.0-1.0)
    input_image = input_image / 255.0
    # チャンネル順の変換 (BGR -> RGB)
    input_image = input_image[:, :, ::-1]
    # 軸の追加 (HWC -> NCHW: Batch, Channel, Height, Width)
    input_image = np.transpose(input_image, (2, 0, 1)) # HWC to CHW
    input_image = np.expand_dims(input_image, 0)      # CHW to NCHW (add batch dimension)
    
    # ONNXモデルの入力形式に合わせてfloat32型に変換
    input_image = input_image.astype(np.float32)
    return input_image

# --- 後処理関数 (YOLOv5の出力例を想定) ---
def postprocess(frame, output):
    # YOLOv5の出力形式を仮定: [batch_size, num_boxes, 5 + num_classes]
    # 5: [x_center, y_center, width, height, object_confidence]
    # num_classes: クラスごとの信頼度

    boxes = []
    confidences = []
    class_ids = []

    # 出力は [batch_size, num_detections, 5+num_classes] または [batch_size, x, y, 5+num_classes] などの形式
    # ここでは一般的なYOLOv5の出力形式を想定します。モデルによって出力形式は異なります。
    # 例えば、出力が (1, 25200, 85) のような形の場合
    output = output[0] # バッチ次元を削除

    img_height, img_width = frame.shape[:2]
    x_factor = img_width / INPUT_WIDTH
    y_factor = img_height / INPUT_HEIGHT

    for detection in output:
        confidence = detection[4] # object_confidence
        if confidence >= CONF_THRESHOLD:
            # クラスごとの信頼度を取得し、最も高いものを選択
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]

            if (confidence * class_conf) >= CONF_THRESHOLD: # オブジェクトとクラス両方の信頼度を考慮
                center_x = int(detection[0] * x_factor)
                center_y = int(detection[1] * y_factor)
                width = int(detection[2] * x_factor)
                height = int(detection[3] * y_factor)
                
                # バウンディングボックスの左上座標を計算
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                boxes.append([left, top, width, height])
                confidences.append(float(confidence * class_conf)) # ここで両方の信頼度を掛け合わせる
                class_ids.append(class_id)
    
    # 非最大抑制 (NMS) を適用して重複するバウンディングボックスを除去
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    
    if len(indices) > 0:
        indices = indices.flatten()
    else:
        indices = [] # 検出がない場合は空のリストにする

    return boxes, confidences, class_ids, indices

# --- メイン処理 ---
def main():
    cap = cv2.VideoCapture(0) # 0番目のWebカメラを使用 (環境により異なる場合があります)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    frame_count = 0
    start_time = time.time()

    print("Starting real-time object detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # 前処理
        input_data = preprocess(frame)

        # 推論実行
        # ONNX Runtimeの推論は、辞書形式で入力データを渡す
        # output_names は、推論結果として取得したい出力層の名前のリスト
        outputs = session.run(output_names, {input_name: input_data})
        
        # 後処理 (outputs[0] は通常、モデルの単一の主要な出力)
        boxes, confidences, class_ids, indices = postprocess(frame.copy(), outputs[0])

        # 結果の描画
        for i in indices:
            box = boxes[i]
            left, top, width, height = box
            
            # バウンディングボックスとラベルを描画
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
            
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # FPS計算
        frame_count += 1
        if frame_count >= 30: # 30フレームごとにFPSを更新
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()
        
        # 結果表示
        cv2.imshow('Real-time Object Detection', frame)

        # 'q' キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()