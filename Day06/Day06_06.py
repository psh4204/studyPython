## 동영상 딥러닝 (SSD) ##
import cv2
import numpy as np
import random


## 함수
### 딥러닝 기반의 사물 인식
def ssdNet(image):
    CONF_VALUE = 0.5  # 컨피던스 값 조정
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = cv2.dnn.readNetFromCaffe("./MibileNetSSD/MobileNetSSD_deploy.prototxt.txt",
                                   "./MibileNetSSD/MobileNetSSD_deploy.caffemodel")
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_VALUE:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (255,255,255), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return image


def snapshot(f) :
    cv2.imwrite('save' + str(random.randint(11111,99999)) + '.png', f)


## 전역
# filename = './faces.mp4'
# capture = cv2.VideoCapture(filename)
frameCount = 0
capture = cv2.VideoCapture(0)
s_factor = 0.8  # 화면크기 비율 ( 조절 가능 )

## 메인

### 동영상 출력 ###
while True:
    ret, frame = capture.read()
    if not ret:
        break

    ## 동영상 딥러닝(SSD) ##
    frameCount += 1
    if frameCount % 5 == 0:
        frame = cv2.resize(frame, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)
        frame = ssdNet(frame)
        cv2.imshow('Video', frame)

    ## 키 입력 ##
    key = cv2.waitKey(20)
    if key == 27:  # ESC 키
        break
    if key == ord('c') or key == ord('C'):
        snapshot(frame)
capture.release()
cv2.destroyAllWindows()
