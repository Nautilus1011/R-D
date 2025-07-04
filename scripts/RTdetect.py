import cv2
import numpy as np
import onnxruntime
import time

model_path = "/home/melnuts/Workspace/repo/R-D/models/yolov5s.onnx"
names_path = "/home/melnuts/Workspace/repo/R-D/models/coco.names"


""" ONNX Runtimeセッションの初期化 """
try:
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
except Exception as e:
    print(f"ONNXモデルのロード中にエラーが発生しました:{e}")
    exit()

""" クラスの読み込み """
try:
    with open(names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"クラス名ファイル '{names_path}' が正常にロードされました")
except FileNotFoundError:
    print(f"エラー：クラス名ファイル '{names_path}' が見つかりません")
    exit()
except Exception as e:
    print(f"クラス名ファイルのロード中にエラーが発生しました：{e}")
    exit()

""" ウェブカメラの初期化 """
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("エラー：ウェブカメラを開けませんでした。")
    exit()

print("\nリアルタイム物体検出開始")


prev_frame_time = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームを読み込めませんでした")
            break

        resized = cv2.resize(frame, (640, 640))
        input_img = resized.transpose(2,0,1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_img, axis=0)

        # session.run()は複数の出力がある場合に備えてリスト形式で結果を返す
        # 今回の場合は[output_name]と一つだけを出力として指定
        # output[0]にはYOLOv5モデルの推論結果が格納される
        # 具体的には次のデータが格納される(バッチサイズ, アンカーボックスの総数, 検出候補の種類と数)
        # 今回の場合
        # 入力データが画像1枚なのでバッチサイズは１
        # yolov5のアンカーボックスは25200で固定
        # 検出候補の種類と数は85となっている
        # つまり、outputsは行数25200, 列数85の2次元配列となっている
        # 85の内訳は以下の通り
        # det[0] = x, det[1] = y, det[2] = w, det[3] = h, det[4] = conf, det[5] ~ det[84] = class_scores
        # 0~3まではバウンディングボックスに関する情報
        # 4は正確性
        # 5以降はcocoデータセット80種それぞれのスコア
        outputs = session.run([output_name], {input_name: input_tensor})[0]
        


        print(outputs[0].shape)


        detected_objects_in_frame = []

        # outputs[0]は2次元配列
        # detは行データ
        # 各行を取り出して処理していく
        for det in outputs[0]:
            conf = det[4]
            if conf > 0.4:
                class_id = int(det[5])
                object_name = class_names[class_id] if class_id < len(class_names) else f"Unknown({class_id})"
                detected_objects_in_frame.append(f"{object_name} (信頼度：{conf:.2f})")
            
        if detected_objects_in_frame:
            current_time = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{current_time}] 検出された物体：{', '.join(detected_objects_in_frame)}")
        else:
            current_time = time.strftime("%H:%M:%S", time.localtime())
        
        curr_frame_time = time.time()
        fps = 1 / (curr_frame_time - prev_frame_time)
        prev_frame_time = curr_frame_time

except KeyboardInterrupt:
    print("\nCtrl+C が押されました。リアルタイム物体検出を修了します。")
except Exception as e:
    print(f"\n予期せぬエラーが発生しました:{e}")

finally:
    """ 処理終了 """
    cap.release()
    print("\nカメラリソースを解放しました。プログラムを終了します。")