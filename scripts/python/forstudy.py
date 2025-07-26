import cv2
import numpy as np
import onnxruntime

model_path = "../../models/onnx_weights/yolov5s.onnx"
names_path = "../../models/coco.names"

session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

with open(names_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (640,640))
        




"""

input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

print("Input name :", input_name)
print("Input shape :", input_shape)
print("Input type :", input_type)

print("Output name :", output_name)

"""