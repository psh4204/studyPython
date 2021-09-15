# 영상(정지,웹캠, 유튜브) 처리 툴
# ===================================
# 기능
# - 좌측화면은  원본 출력(720x까지)
# - 중간화면은 처리 출력(1초에1번)
# - 우측화면은 정보 출력(객체인식이미지까지)
# 사진, 동영상, 웹캠, 유튜브까지 가능하게끔.
# ===================================
# -*-coding:utf-8-*-
import sys
import math
from tkinter import *
from tkinter.colorchooser import *
from tkinter.simpledialog import *
from tkinter.filedialog import *
import cv2
import numpy as np
import os
import random
import threading
import pafy, youtube_dl


# # ============ 함수 =================
# # # -------------- 공통함수 -------------------
def mainThread():
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global cvInPhoto, cvOutPhoto
    global canvas_hist, paper_hist, canvas_hist
    global sx, sy, ex, ey, boxLine, status
    # # # 메뉴창 선언
    mainMenu = Menu(window)  # 메뉴 창
    window.config(menu=mainMenu)  # 세부설정(메뉴 창)
    # # # # 메뉴창_파일
    fileMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="파일", menu=fileMenu)
    fileMenu.add_command(label="열기", command=openImage)
    fileMenu.add_command(label="저장", command=saveImage)
    fileMenu.add_separator()
    fileMenu.add_command(label="종료", command=None)

    #### 메뉴창_화소점처리
    youtubeMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="Youtube", menu=youtubeMenu)
    youtubeMenu.add_command(label="출력", command=yotubeDisplayThreading)

    # # # 화면 선언
    # # # # 캔버스 (Output Image)
    canvas = Canvas(window, height=m_outH + 255, width=m_outW + 255, bg='black')  # 캔버스 위젯
    paper = PhotoImage(width=500, height=500)  # paper 초기화
    canvas.create_image((500 / 2, 500 / 2), image=paper,
                        state='normal')  # 캔버스 중앙에 paper 그리기
    # # # # 캔버스 (Histogram)
    canvas_hist = Canvas(window, height=255, width=255, bg='black')
    paper_hist = PhotoImage(width=255, height=255)  # paper 초기화
    canvas_hist.create_image((255 / 2, 255 / 2), image=paper,
                             state='normal')  # 캔버스 중앙에 paper 그리기
    ##### pack
    canvas_hist.grid(row=0, column=1, sticky="s")
    canvas.grid(row=0, column=2, sticky="n")

    # # # 라디오 버튼 선언
    radVar = IntVar()
    r1 = tkinter.Radiobutton(window, text="일반", variable=radVar, value=1)
    r2 = tkinter.Radiobutton(window, text="평활화", variable=radVar, value=2)
    ##### pack
    r1.place(x=0, y=20)
    r2.place(x=0, y=40)

    window.mainloop()


def malloc3D(h, w, init=0):
    # init =0 <- 디폴트 파라미터 선언
    global RGB
    # 지역변수
    memory = [[[init for _ in range(w)] for _ in range(h)]
              for _ in range(RGB)]  # RGB 3차원 리스트 선언
    return memory


def malloc3D_double(h, w, init=0.0):  # *** 수정바람
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
    # 지역변수
    memory = [[[init for _ in range(w)] for _ in range(h)]
              for _ in range(RGB)]  # 리스트 함축
    return memory


def openImage():
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global cvInPhoto, cvOutPhoto
    global sx, sy, ex, ey, boxLine, status
    # 파일 선택하고 크기 계산
    ## 파일 이름으로 파일 찾기
    filename = askopenfilename(parent=window,
                               filetypes=(("컬러 파일", "*.jpg *.png *.bmp *tiff *.tif"), ("모든파일", "*.*")))
    ## CV용 개체생성
    cvInPhoto = cv2.imread(filename)

    ## 영상 크기 알아내기
    # m_inH = cvInPhoto.shape[0]
    # m_inW = cvInPhoto.shape[1]
    m_inH, m_inW = cvInPhoto.shape[:2]  # [0:2]0~1까지
    # 메모리 할당
    m_InputImage = malloc3D(m_inH, m_inW)
    # 파일 --> 메모리
    ## OpenCV 는 BGR 순으로 출력
    for i in range(m_inH):
        for k in range(m_inW):
            m_InputImage[RR][i][k] = cvInPhoto.item(i, k, BB)
            m_InputImage[GG][i][k] = cvInPhoto.item(i, k, GG)
            m_InputImage[BB][i][k] = cvInPhoto.item(i, k, RR)
    print(m_InputImage[RR][100][100], m_InputImage[GG]
    [100][100], m_InputImage[BB][100][100])
    equalImage()


##### 이미지 저장
def saveImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
    # 빈 OpenCV 개체 생성
    saveCvPhoto = np.zeros((m_outH, m_outW, 3), np.uint8)  # 넘파이배열 사용
    # 출력리스트 --> 넘파이 배열
    ## BGR 순서
    for i in range(m_outH):
        for k in range(m_outW):
            tp = tuple(
                ([m_OutputImage[BB][i][k], m_OutputImage[GG][i][k], m_OutputImage[RR][i][k]]))  # tuple(): 튜플로 만들어주는 함수
            saveCvPhoto[i, k] = tp

    saveFp = asksaveasfile(parent=window, mode='wb', defaultextension='.',
                           filetypes=(("컬러 파일", "*.jpg *.png *.bmp *tiff *.tif"), ("모든파일", "*.*")))
    cv2.imwrite(saveFp.name, saveCvPhoto)
    print("Save")


# # # -------------- 영상처리함수 -------------
# # # # 동일 이미지
def equalImage(): 
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global cvInPhoto, cvOutPhoto
    global sx, sy, ex, ey, boxLine, status
    if not m_InputImage:
        messagebox.showinfo("알림", "이미지가 입력되지 않았습니다.")
        return
    # (중요!) 출력 영상 크기 결정 --> 알고리즘에 따름
    m_outH = m_inH
    m_outW = m_inW
    # 출력 메모리 할당
    m_OutputImage = malloc3D(m_outH, m_outW)
    ## 진짜 영상처리 알고리즘
    for rgb in range(RGB):
        for h in range(m_inH):
            for w in range(m_inW):
                m_OutputImage[rgb][h][w] = m_InputImage[rgb][h][w]
    displayImage()


# OUTPUT 디스플레이 #TODO : 확대에서 막힘
def displayImage():
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global m_reH, m_reW
    global m_histImage
    global cvInPhoto, cvOutPhoto
    global sx, sy, ex, ey, boxLine
    # 캔버스
    ## 캔버스 초기화(이전것 없애기)
    if canvas != None:
        canvas.destroy()
    # 히스토그램 출력
    histogramImage()
    ## 사이즈 일정하게 ( 640 x 360 크기 안으로 )
    tempOutputImage = None
    reH = m_outH
    reW = m_outW
    maxSize = 640 * 360
    reScale = 1
    ## 빠른출력(C++식, 더블버퍼, 빠름 )
    rgbString = ""
    ### Output사이즈가 크면 (줄이기)
    if maxSize <= reH * reW:
        print("축소")
        reScale = int(math.sqrt((reH * reW) / maxSize))
        if int(reScale) <= 0:
            reScale = 1
        reH = int(reH / reScale)
        reW = int(reW / reScale)
        tempOutputImage = malloc3D(reH, reW)
        for rgb in range(RGB):
            for i in range(reH):
                for k in range(reW):
                    m = m_OutputImage[rgb][int(i * reScale)][int(k * reScale)]
                    tempOutputImage[rgb][i][k] = m

    ### 사이즈가 작으면 (양선형 보간으로 키우기)
    elif maxSize > reH * reW:
        print("양선형확대")
        reScale =  int(math.sqrt(maxSize / (reH * reW)))
        if int(reScale) <= 0:
            reScale = 1
        reH = int(m_outH * reScale)
        reW = int(m_outW * reScale)
        tempImage = malloc3D(m_outH, m_outW)
        tempOutputImage = malloc3D(reH, reW)
        for rgb in range(RGB):
            for i in range(m_outH):
                for k in range(m_outW):
                    tempImage[rgb][i][k] = m_OutputImage[rgb][i][k]

        r_H, r_W, s_H, s_W = [0.0] * 4
        i_W, i_H, v = [0] * 3
        for rgb in range(RGB):
            for i in range(reH):
                for k in range(reW):
                    r_H = i / reScale
                    r_W = k / reScale

                    i_H = int(math.floor(r_H))  # 내림.celi = 올림
                    i_W = int(math.floor(r_W))

                    s_H = r_H - i_H
                    s_W = r_W - i_W

                    if (i_H < 0 or i_H >= (m_outH - 1) or i_W < 0 or i_W >= (m_outW - 1)):
                        tempOutputImage[rgb][i][k] = 255
                    else:
                        C1 = tempImage[rgb][i_H][i_W]  # A
                        C2 = tempImage[rgb][i_H][i_W + 1]  # B
                        C3 = tempImage[rgb][i_H + 1][i_W + 1]  # C
                        C4 = tempImage[rgb][i_H + 1][i_W]  # D
                        v = int(C1 * (1 - s_H) * (1 - s_W) + C2 * s_W *
                                (1 - s_H) + C3 * s_W * s_H + C4 * (1 - s_W) * s_H)
                        tempOutputImage[rgb][i][k] = v
    print("scale :", reScale)
    print(reH, "x", reW)
    for i in range(reH):
        tmpStr = ""  # 한 라인
        for k in range(reW):
            r = tempOutputImage[RR][i][k]
            g = tempOutputImage[GG][i][k]
            b = tempOutputImage[BB][i][k]
            # 'rgb ' 로 넣기 (마지막 띄어쓰기로 구분처리(구분처리 안하면 못읽음))
            tmpStr += "#%02x%02x%02x " % (r, g, b)
        # '{rgb} ' 로 넣기 (마지막 띄어쓰기로 구분처리(구분처리 안하면 못읽음))
        rgbString += "{" + tmpStr + "} "

    ## 윈도우 화면 크기 딱 맞추기
    ### oepnCV는 geometry 가 반대이다.
    window.geometry(str(reW + 355) + "x" + str(reH + 20))
    canvas = Canvas(window, height=reH, width=reW)  # 캔버스 위젯
    paper = PhotoImage(height=reH, width=reW)  # paper 초기화
    canvas.create_image((reW / 2, reH / 2), image=paper,
                        state='normal')  # 캔버스 중앙에 paper 그리기
    paper.put(rgbString)
    # canvas 출력
    canvas_hist.grid(row=0, column=1, sticky="s")
    canvas.grid(row=0, column=2, sticky="n")

    # 상태창 설정
    status.configure(text=str(m_inW) + "x" + str(m_inH) + '    ' + filename)


# HISTOGRAM 디스플레이
def displayHistogram():
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global canvas_hist, paper_hist, canvas_hist
    global m_histImage
    global cvInPhoto, cvOutPhoto
    global sx, sy, ex, ey, boxLine
    # 캔버스
    ## 캔버스 초기화(이전것 없애기)
    if canvas_hist != None:
        canvas_hist.destroy()
    ## 윈도우 화면 크기 딱 맞추기
    ### oepnCV는 geometry 가 반대이다.
    canvas_hist = Canvas(window, height=256, width=256)  # 캔버스 위젯
    paper_hist = PhotoImage(height=256, width=256)  # paper 초기화
    canvas_hist.create_image((256 / 2, 256 / 2), image=paper_hist,
                             state='normal')  # 캔버스 중앙에 paper 그리기
    ## 빠른출력(C++식, 더블버퍼, 빠름)
    rgbString = ""
    for i in range(256):
        tmpStr = ""  # 한 라인
        for k in range(256):
            r = m_histImage[RR][i][k]
            g = m_histImage[GG][i][k]
            b = m_histImage[BB][i][k]
            # 'rgb ' 로 넣기 (마지막 띄어쓰기로 구분처리(구분처리 안하면 못읽음))
            tmpStr += "#%02x%02x%02x " % (r, g, b)
        # '{rgb} ' 로 넣기 (마지막 띄어쓰기로 구분처리(구분처리 안하면 못읽음))
        rgbString += "{" + tmpStr + "} "
    paper_hist.put(rgbString)
    # canvas pack
    canvas_hist.grid(row=0, column=1)
    # 상태창 설정


#### 히스토그램 출력 (BGR)
def histogramImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
    global canvas_hist, paper_hist, canvas_hist
    global m_histImage
    if not m_OutputImage:
        messagebox.showinfo("알림", "이미지가 입력되지 않았습니다.")
        return
    print("[히스토그램출력]")
    # 히스토그램 크기
    outH = 256
    outW = 256
    # 출력 메모리 할당
    m_histImage = malloc3D(outH, outW)
    # 히스토그램 데이터 초기화
    reSize = outH * outW
    LOW = 0
    HIGH = 255
    histR = [0 for _ in range(256)]
    histG = [0 for _ in range(256)]
    histB = [0 for _ in range(256)]
    valueR, valueG, valueB = [0] * 3
    print(valueR, valueG, valueB)
    # 빈도수 조사
    for i in range(m_outH):
        for k in range(m_outW):
            valueR = m_OutputImage[RR][i][k]
            valueG = m_OutputImage[GG][i][k]
            valueB = m_OutputImage[BB][i][k]
            histR[valueR] += 1
            histG[valueG] += 1
            histB[valueB] += 1
    # 정규화
    minR, minG, minB = [0] * 3
    maxR, maxG, maxB = [0] * 3
    difR, difG, difB = [0] * 3
    print(minR, minG, minB)
    print(maxR, maxG, maxB)
    print(difR, difG, difB)
    for i in range(256):
        if (histR[i] <= minR):
            minR = histR[i]
        if (histR[i] >= maxR):
            maxR = histR[i]
        if (histG[i] <= minG):
            minG = histG[i]
        if (histG[i] >= maxG):
            maxG = histG[i]
        if (histB[i] <= minB):
            minB = histB[i]
        if (histB[i] >= maxB):
            maxB = histB[i]
    difR = maxR - minR
    difG = maxG - minG
    difB = maxB - minB
    print(difR, difG, difB)
    scaleHistR = [255 for _ in range(256)]
    scaleHistG = [255 for _ in range(256)]
    scaleHistB = [255 for _ in range(256)]
    # 정규화 된 히스토그램
    for i in range(256):
        scaleHistR[i] = int((histR[i] - minR) * HIGH / difR)
        scaleHistG[i] = int((histG[i] - minG) * HIGH / difG)
        scaleHistB[i] = int((histB[i] - minB) * HIGH / difB)
    # 정규화된 히스토그램 출력
    OutImageR = [255 for _ in range(reSize)]
    OutImageG = [255 for _ in range(reSize)]
    OutImageB = [255 for _ in range(reSize)]
    for i in range(outH):
        for k in range(scaleHistR[i]):
            OutImageR[outW * (outH - k - 1) + i] = 0
        for k in range(scaleHistG[i]):
            OutImageG[outW * (outH - k - 1) + i] = 0
        for k in range(scaleHistB[i]):
            OutImageB[outW * (outH - k - 1) + i] = 0
    histNum = 0
    # BGR로 출력
    for i in range(outH):
        for k in range(outW):
            m_histImage[RR][i][k] = OutImageB[histNum]
            m_histImage[GG][i][k] = OutImageG[histNum]
            m_histImage[BB][i][k] = OutImageR[histNum]
            histNum += 1
    displayHistogram()


### OpenCV용 함수
#### 결과를 화면출력으로 보내기
def cv2OutImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
    global cvInPhoto, cvOutPhoto
    # 크기 알아내기
    m_outH, m_outW = cvOutPhoto.shape[:2]  # [0:2]0~1까지
    # 메모리 할당
    m_OutputImage = malloc3D(m_outH, m_outW)
    # 3차원 스케일
    # if not m_OutputImage[RR]:
    if cvOutPhoto.ndim > 2:  # ndim : 넘파이 차원수 확인
        for i in range(m_outH):
            for k in range(m_outW):
                m_OutputImage[RR][i][k] = cvOutPhoto.item(i, k, BB)
                m_OutputImage[GG][i][k] = cvOutPhoto.item(i, k, GG)
                m_OutputImage[BB][i][k] = cvOutPhoto.item(i, k, RR)
    # 단일 스케일
    # elif m_OutputImage[RR]:
    else:
        for i in range(m_outH):
            for k in range(m_outW):
                m_OutputImage[RR][i][k] = cvOutPhoto.item(i, k)
                m_OutputImage[GG][i][k] = cvOutPhoto.item(i, k)
                m_OutputImage[BB][i][k] = cvOutPhoto.item(i, k)


# Youtube 영상출력 
def youtubeDisplay():
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global cvInPhoto, cvOutPhoto
    global sx, sy, ex, ey, boxLine, status
    global url
    print("[Youtube_영상출력]")
    url = askstring("입력창", "출력할 유튜브 영상을 url을 입력하시오")
    if not url or url == "":
        return
    ytVideo = pafy.new(url)
    frameCount = 0
    best = ytVideo.getbest(preftype='mp4')  # 'webm','3gp'
    capture = cv2.VideoCapture(best.url)
    title = ytVideo.title
    s_factor = 0.8  # 화면크기 비율 ( 조절 가능 )
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        ## 동영상 딥러닝(SSD) ##
        frameCount += 1
        if frameCount % 5 == 0:
            frame = cv2.resize(frame, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)
            # frame = ssdNet(frame)
            # 화면출력
            cv2.imshow("Youtube", frame)
        key = cv2.waitKey(25)
        ## 키입력
        if key == 27:  # Esc
            cvOutPhoto = frame
            filename = title
            cv2OutImage_UCC()
            displayImage()
            break
        if key == ord('c') or key == ord('C'):
            filename = title
            snapshot(frame)
    capture.release()
    cv2.destroyAllWindows()


#### Youtube영상출력 쓰레드
def yotubeDisplayThreading():
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global cvInPhoto, cvOutPhoto
    global sx, sy, ex, ey, boxLine
    thread_Youtube = threading.Thread(target=youtubeDisplay(), daemon=True)
    thread_Youtube.start()
    thread_Youtube.join()


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
                          (255, 255, 255), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return image


def snapshot(f):
    global filename
    cv2.imwrite('save' + str(random.randint(11111, 99999)) + '.png', f)


#### 웹캠인식_SSD Todo : 타이틀, 모드설정
def displayUCC():
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global cvInPhoto, cvOutPhoto
    global sx, sy, ex, ey, boxLine
    global lock
    frameCount = 0
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # captureDevice = camera ## 카메라 캡쳐를 백에드에서 시키기
    s_factor = 1  # 화면크기 비율 ( 조절 가능 )
    while True:
        ret, frame = capture.read()
        print(ret)
        if not ret:
            break
        ## 동영상 딥러닝(SSD) ##
        frameCount += 1
        if frameCount % 10 == 0:
            frame = cv2.resize(frame, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)
            cvOutPhoto = ssdNet(frame)
            frame = cvOutPhoto
            cv2.imshow('Video', frame)
            ## 화면출력
            cv2OutImage()
            displayImage()

        ## 키 입력 ##
        key = cv2.waitKey(6)
        if key == 27:  # ESC 키
            break
        if key == ord('c') or key == ord('C'):
            snapshot(frame)
    capture.release()
    print("[동영상인식_SSD]")
    cv2.destroyAllWindows()


def displayUCCThreading():
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global cvInPhoto, cvOutPhoto
    global sx, sy, ex, ey, boxLine
    thread_UCC = threading.Thread(target=displayUCC(), daemon=True)
    thread_UCC.start()
    thread_UCC.join()


#### 결과를 화면출력으로 보내기
def cv2OutImage_UCC():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
    global cvInPhoto, cvOutPhoto
    # 크기 알아내기
    m_outH, m_outW = cvOutPhoto.shape[:2]  # [0:2]0~1까지
    # 메모리 할당
    m_OutputImage = malloc3D(m_outH, m_outW)
    # 3차원 스케일
    # if not m_OutputImage[RR]:
    if cvOutPhoto.ndim > 2:  # ndim : 넘파이 차원수 확인
        for i in range(m_outH):
            for k in range(m_outW):
                m_OutputImage[RR][i][k] = cvOutPhoto.item(i, k, BB)
                m_OutputImage[GG][i][k] = cvOutPhoto.item(i, k, GG)
                m_OutputImage[BB][i][k] = cvOutPhoto.item(i, k, RR)
    # 단일 스케일
    # elif m_OutputImage[RR]:
    else:
        for i in range(m_outH):
            for k in range(m_outW):
                m_OutputImage[RR][i][k] = m_InputImage[RR][i][k] = cvOutPhoto.item(i, k)
                m_OutputImage[GG][i][k] = m_InputImage[GG][i][k] = cvOutPhoto.item(i, k)
                m_OutputImage[BB][i][k] = m_InputImage[BB][i][k] = cvOutPhoto.item(i, k)
    m_inH, m_inW = m_outH, m_outW


#### Todo : 위젯설정 (영상두개)

#### ToDo : OpenCV 모드 설정

#### ToDo : Output 그림크기 일정하게 (작으면 키우고, 크면 작게)


# # ========== 전역변수 ================
# # # 윈도우창, (OutPut)캔버스, 이미지용 화면
window, canvas, paper = None, None, None
canvas_hist, paper_hist, canvas_hist = None, None, None
# # # # Input Canvas
inCanvas = None
# # # 파일 주소
filename = ""
m_InputImage, m_OutputImage = None, None
m_histImage = None
m_inH, m_inW, m_outH, m_outW = [0] * 4
RGB, RR, GG, BB = 3, 0, 1, 2
# # # OpenCV 변수
cvInPhoto, cvOutPhoto = None, None
# # # 마우스 관련
sx, sy, ex, ey = [-1] * 4
boxLine = None
# # # 동영상 관련
# filename = './faces.mp4'
# capture = cv2.VideoCapture(filename)
# # # 유튜브 관련
url = ""
window = Tk()
window.title("영상처리기 Ver 0.2")  # 타이틀
window.geometry(str(255 + 355) + "x" + str(255 + 40))  # 윈도우창 크기
window.resizable(width=False, height=False)  # 사이즈 조정여부
# # # 모니터 사이즈 구하기
monitor_height = window.winfo_screenheight()
monitor_width = window.winfo_screenwidth()
# # # 상태바
status = Label(window, text='이미지 정보', bd=1, relief=SUNKEN, anchor=W)  # 상태바
status.grid(row=1, column=2, sticky="s")
# # # 라디오버튼 타이틀
radTitle = Label(window, text='[ OpenCV모드 ]', bd=1)
radTitle.grid(row=0, column=0, sticky="n")
# # # 히스토그램 타이틀
histTitle = Label(window, text='히스토그램', bd=1, relief=SUNKEN, anchor=W)
histTitle.grid(row=1, column=1, sticky="n")
# # ============ 메인 =================
if __name__ == '__main__':
    thread_main = threading.Thread(target=mainThread(), args=())
    thread_main.daemon = True
    thread_main.start()
