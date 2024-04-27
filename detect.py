import numpy as np
import cv2
import serial


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    ratioh, ratiow = frameHeight / 640, frameWidth / 640
    confidences = []
    boxes = []
    for detection in outs:
        confidence = detection[5]
        if confidence > 0.5 and detection[4] > 0.5:
            center_x = int(detection[0] * ratiow)
            center_y = int(detection[1] * ratioh)
            width = int(detection[2] * ratiow)
            height = int(detection[3] * ratioh)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            confidences.append(float(detection[4]))
            boxes.append([left, top, width, height])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    res = []
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        res.append([left + width / 2, top + height / 2, confidences[i]])
        cv2.circle(frame, (int(left + width / 2), int(top + height / 2)), 4, (0, 0, 255))
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 0, int((400*i)//255)), thickness=4)

    return res

import time
def detect(image, net):
      # 读取图片
    # image = cv2.imread(r"C:\Users\Primo\Desktop\mmexport1695388346969.jpg")
    image = cv2.resize(image, (640, 640))
    blob = cv2.dnn.blobFromImage(image, 1 / 255, size=(640, 640))
    net.setInput(blob)  # 设置模型输入
    out = net.forward()  # 推理出结果
    out = out[0]
    res = postprocess(image, out)
    # print(np.shape(image))
    # print(res)
    cv2.imshow("abc", image)
    cv2.waitKey(0)
    # ser=serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
    #
    print(res)
    if len(res)>1:
        min = 0
        b = 0
        for i in range(len(res)):
            if res[i][2]<min:
                min = res[i][2]
                b = i
        res = [res[b]]
    print(res)
    # ser.write(chr(len(res)).encode())
    # try:
    #     ser.write(chr(int(res[0][0]/5)).encode())
    #     ser.write(chr(int(res[0][1]/5)).encode())
    # except:
    #     ser.write(chr(255).encode())
    #     ser.write(chr(255).encode())
    # ser.close()


# ser.write()

# ser.write()
cap = cv2.VideoCapture(1)
_, image = cap.read()
detect(image, net=cv2.dnn.readNetFromONNX("best.onnx"))
# time.sleep(2)
