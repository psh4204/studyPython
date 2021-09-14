import math
from tkinter import *
from tkinter.colorchooser import *
from tkinter.simpledialog import *
from tkinter.filedialog import *
import cv2
import numpy as np
import os
import random

## 함수===========================================================
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

def displayUCC():
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global sx, sy, ex, ey, boxLine
    global frameCount,capture,s_factor
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


## 전역=====================================================
# filename = './faces.mp4'
# capture = cv2.VideoCapture(filename)
frameCount = 0
capture = cv2.VideoCapture(0)
s_factor = 0.8  # 화면크기 비율 ( 조절 가능 )
### 윈도우창, 캔버스, 이미지용 화면
window, canvas, paper = None, None, None
### 파일 주소
filename = ""
m_InputImage, m_OutputImage = None, None
m_inH, m_inW, m_outH, m_outW = [0] * 4
RGB, RR, GG, BB = 3, 0, 1, 2
### OpenCV 변수
cvInPhoto, cvOutPhoto = None, None
### 마우스 관련
sx, sy, ex, ey = [-1] * 4
boxLine = None

## 메인=====================================================
window = Tk()
window.title("컬러 영상처리 Ver 0.7")  # 타이틀
window.geometry('720x520')  # 윈도우창 크기
window.resizable(width=True, height=True)  # 사이즈 조정여부
### 모니터 사이즈 구하기
monitor_height = window.winfo_screenheight()
monitor_width = window.winfo_screenwidth()
displayUCC()

#### 캔버스
canvas = Canvas(window, height=500, width=500)  # 캔버스 위젯
paper = PhotoImage(width=500, height=500)  # paper 초기화
canvas.create_image((500 / 2, 500 / 2), image=paper,
                    state='normal')  # 캔버스 중앙에 paper 그리기

### 동영상 출력 ###


canvas.pack()
window.mainloop()
