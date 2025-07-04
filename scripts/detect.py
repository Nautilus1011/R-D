import cv2
import numpy as np
import onnxruntime
import os

# モデルとクラス名の読み込み
model_path = "/home/melnuts/Workspace/repo/R-D/models/yolov5s.onnx"
image_path = "/home/melnuts/Workspace/repo/R-D/data/images/test001.jpg"
save_path = "/home/melnuts/Workspace/repo/R-D/runs/detect"

session = onnxruntime.InferenceSession(model_path)
# onnxファイル内部に既に定義されている名前を取得
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 入力画像読み込みと前処理
image = cv2.imread(image_path)
resized = cv2.resize(image, (640, 640))
input_img = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
input_tensor = np.expand_dims(input_img, axis=0)

# 推論 
# onnx runtimeでは、モデルに入力・出力を渡すとき
# どの名前の入力に、どのテンソルを渡すか, どの名前の出力を取り出すか
# などを明示的に指定しなければならない
outputs = session.run([output_name], {input_name: input_tensor})[0]

print(outputs)

# 結果から検出ボックスを描画(簡易)
for det in outputs[0]:
    conf = det[4]
    if conf > 0.5:
        class_id = int(det[5])
        x, y, w, h = det[0:4]
        x1 = int((x - w/2) * image.shape[1] / 640)
        y1 = int((y - h/2) * image.shape[0] / 640)
        x2 = int((x + w/2) * image.shape[1] / 640)
        y2 = int((x + h/2) * image.shape[0] / 640)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, f"ID:{class_id}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
cv2.imwrite(os.path.join(save_path, "result.jpg"), image)
print("物体検出完了! -> result.jpg に保存しました")