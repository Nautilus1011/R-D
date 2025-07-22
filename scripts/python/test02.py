"""
WSL-UbuntuからUSBカメラを制御するためのプログラムコード
帯域幅の影響か、解像度を下げないとフレームを取得できない。
最大解像度は480x360
"""


import cv2
import os
import datetime

save_directory = "../runs"

camera_ids_to_try = [0, 1, 2, 3]
cap = None
found_camera = False

for cam_id in camera_ids_to_try:
    print(f"カメラID {cam_id} で開いてみます...")
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)

    if cap.isOpened():
        print(f"カメラID {cam_id} を開けました。")

        # --- 解像度設定 ---
        # 320x240で成功したので、まずはこれを維持
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        # --- タイムアウト時間を設定 ---
        # デフォルトは数秒程度。これを例えば10秒 (10000ミリ秒) に延長してみる
        # これを長くしすぎると、フレームが来ない場合にプログラムが長時間停止します
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000) # 10秒に設定
        print(f"フレーム読み込みタイムアウトを {cap.get(cv2.CAP_PROP_READ_TIMEOUT_MSEC)} ms に設定しました。")

        found_camera = True
        break
    else:
        print(f"カメラID {cam_id} は開けませんでした。")

if not found_camera:
    print("エラー：どのカメラIDも開けませんでした。")
    print("詳細なトラブルシューティングのヒントは以前のメッセージを参照してください。")
    exit()

try:
    print("フレームを取得中...")
    ret, frame = cap.read()

    if ret:
        os.makedirs(save_directory, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"capture_{timestamp}.jpg"
        save_path = os.path.join(save_directory, file_name)
        cv2.imwrite(save_path, frame)
        print(f"画像を保存しました: {save_path}")
    else:
        print("エラー：カメラからフレームを取得できませんでした。")
        print("ウェブカメラのLEDは点灯していても、データストリームが確立されていない可能性があります。")
        print("WSL2では、`usbipd attach` の後の接続が不安定になることがあります。")
        print("PowerShell (管理者) で `usbipd detach` -> `usbipd attach` を試してみてください。")

except Exception as e:
    print(f"予期せぬエラーが発生しました: {e}")

finally:
    if cap is not None and cap.isOpened():
        cap.release()
        print("カメラリソースを解放しました。")
    print("プログラムを終了します。")