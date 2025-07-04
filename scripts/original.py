import cv2
import numpy as np
import onnxruntime
import time
import os

model_path = "/home/melnuts/Workspace/repo/R-D/models/yolov5s.onnx"
names_path = "/home/melnuts/Workspace/repo/R-D/models/coco.names"

try:
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_input()[0].name
    output_name = session.get_output()[0].name
except Exception as e:
    print(f"ONNXモデルのロードに失敗:{e}")
    exit()

try:
    with open(names_path, 'r')