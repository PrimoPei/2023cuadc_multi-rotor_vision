import time
import subprocess
import detect
import serial
import cv2

net = cv2.dnn.readNetFromONNX("best.onnx")
cap = cv2.VideoCapture(0)
while True:
	ret, image = cap.read()
	detect.detect(image, net)
	time.sleep(0.1)
