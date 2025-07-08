import cv2
import os

savePath = "../runs"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("カメラ開けなかった")
    exit()



# 一枚の画像を表示するコード
# """

ret, frame = cap.read()

if not ret:
    print("can't get frame")
else:
    # cv2.imwrite(os.path.join(savePath, "output.jpg"), frame)
    # print("保存完了")
    # print("frame.shape:", frame.shape)
    # print("frame.mean():", frame.mean())

    cv2.imshow("取得画像", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# """

# リアルタイムで連続して表示するコード
"""

while True:
    ret, frame = cap.read()
    if not ret:
        print("can't get frame")
        break

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

"""
cap.release()
cv2.destroyAllWindows()