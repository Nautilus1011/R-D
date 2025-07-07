import cv2
import numpy as np
import onnxruntime
import time
import os

model_path = "../models/yolov5s.onnx"
names_path = "../models/coco.names"

try:
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_input()[0].name
    output_name = session.get_output()[0].name
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
