import cv2
import numpy as np
import onnxruntime


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
    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    results = []
    while detections:
        best = detections.pop(0)
        results.append(best)
        detections = [d for d in detections if iou(best[0], d[0]) < iou_threshold]
    return results

# ------------ モデルとラベル読み込み ------------
model_path = "../../models/onnx_weights/yolov5s.onnx"
names_path = "../../models/coco.names"

session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

with open(names_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# ------------ カメラ起動 ------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラが開けませんでした")
    exit()

# ------------ メインループ ------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 前処理
        resized = cv2.resize(frame, (640, 640))
        rgb_frame = resized[:, :, ::-1]
        input_img = rgb_frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_img, axis=0)

        # 推論
        outputs = session.run([output_name], {input_name: input_tensor})[0]

        # 候補抽出
        candidates = []
        for det in outputs[0]:
            conf = det[4]
            if conf > 0.2:
                x_center, y_center, width, height = det[0], det[1], det[2], det[3]
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                class_id = int(det[5])
                candidates.append(((x1, y1, x2, y2), conf, class_id))

        # NMSで重複除去
        results = non_max_suppression(candidates, iou_threshold=0.5)

        # 結果を表示
        detected_objects = []
        for box, conf, class_id in results:
            object_name = class_names[class_id] if class_id < len(class_names) else f"Unknown({class_id})"
            detected_objects.append(f"{object_name} (信頼度：{conf:.2f})")

        if detected_objects:
            print("検出された物体：" + ", ".join(detected_objects))

except KeyboardInterrupt:
    print("終了します")

finally:
    cap.release()
