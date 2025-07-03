import cv2
import os

save_path = "/home/melnuts/Workspace/repo/R-D/runs"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラが開けませんでした")
    exit()

ret, frame = cap.read()

if ret:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(os.path.join(save_path, "capture.jpg"), frame)
    print(f"画像を保存しました:{save_path}")
else:
    print("フレームの取得に失敗しました")

cap.release()