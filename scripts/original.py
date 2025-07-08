import cv2
import numpy as np
import onnxruntime
import time
import os

model_path = "/home/melnuts/Workspace/repo/R-D/models/yolov5s.onnx"
names_path = "/home/melnuts/Workspace/repo/R-D/models/coco.names"

try:
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
except Exception as e:
    print(f"ONNXモデルのロードに失敗:{e}")
    exit()

try:
    with open(names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()] 
except FileNotFoundError:
    print("エラー：クラス名ふぁるが見つからない")
    exit()
except Exception:
    print("クラス名ファイルのロード中にエラーが発生")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("エラー：ウェブカメラを開けませんでした")
    exit()


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームを読み込めなかった")
            break

        
        resized = cv2.resize(frame, (640,640))
        input_img = resized.transpose(2,0,1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_img, axis=0)
        
        outputs = session.run([output_name], {input_name: input_tensor})[0]

        detected_objects_in_frame = []

        for det in outputs[0]:
            conf = det[4]
            if conf > 0.4:
                class_id = int(det[5])
                object_name = class_names[class_id] if class_id < len(class_names) else f"Unknown({class_id})"
                detected_objects_in_frame.append(f"{object_name} (信頼度：{conf:.2f})")

        if detected_objects_in_frame:
            print(detected_objects_in_frame)

except KeyboardInterrupt:
    print("検出フェーズ終了")
except Exception as e:
    print("予期せぬエラー")

finally:
    cap.release()
    print("カメラリソース開放, プログラム終了")