# OpenCV 첫 수업
import math
from tkinter import *
from tkinter.colorchooser import *
from tkinter.simpledialog import *
from tkinter.filedialog import *
import cv2
import numpy as np
import os


## 함수
### 공통함수
def malloc3D(h, w, init=0):
    # init =0 <- 디폴트 파라미터 선언
    global RGB
    # 지역변수
    memory = [[[init for _ in range(w)] for _ in range(h)] for _ in range(RGB)]  # RGB 3차원 리스트 선언
    return memory


def malloc2D_double(h, w):  # *** 수정바람
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
    # 지역변수
    memory = [[0.0 for _ in range(w)] for _ in range(h)]  # 리스트 함축
    return memory


def openImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
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
    print(m_InputImage[RR][100][100], m_InputImage[GG][100][100], m_InputImage[BB][100][100])
    equalImage()


def displayImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
    # 캔버스
    ## 캔버스 초기화(이전것 없애기)
    if canvas != None:
        canvas.destroy()
    ## 윈도우 화면 크기 딱 맞추기
    ### oepnCV는 geometry 가 반대이다.
    window.geometry(str(m_outW) + "x" + str(m_outH))
    canvas = Canvas(window, height=m_outH, width=m_outW)  # 캔버스 위젯
    paper = PhotoImage(height=m_outH, width=m_outW)  # paper 초기화
    canvas.create_image((m_outW / 2, m_outH / 2), image=paper, state='normal')  # 캔버스 중앙에 paper 그리기
    ## 빠른출력(C++식, 더블버퍼, 빠름)
    rgbString = ""
    for i in range(m_outH):
        tmpStr = ""  # 한 라인
        for k in range(m_outW):
            r = m_OutputImage[RR][i][k]
            g = m_OutputImage[GG][i][k]
            b = m_OutputImage[BB][i][k]
            # 'rgb ' 로 넣기 (마지막 띄어쓰기로 구분처리(구분처리 안하면 못읽음))
            tmpStr += "#%02x%02x%02x " % (r, g, b)
        # '{rgb} ' 로 넣기 (마지막 띄어쓰기로 구분처리(구분처리 안하면 못읽음))
        rgbString += "{" + tmpStr + "} "
    paper.put(rgbString)
    # canvas 출력
    canvas.pack()
    # 상태창 설정
    status.configure(text=str(m_inW) + "x" + str(m_inH) + '    ' + filename)


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


### 영상처리함수
#### 동일 이미지
def equalImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
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


#### 반전 이미지
def reverseImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
    # (중요!) 출력 영상 크기 결정 --> 알고리즘에 따름
    m_outH = m_inH
    m_outW = m_inW
    # 출력 메모리 할당
    m_OutputImage = malloc3D(m_outH, m_outW)
    ## 진짜 영상처리 알고리즘
    for rgb in range(RGB):
        for h in range(m_inH):
            for w in range(m_inW):
                m_OutputImage[rgb][h][w] = 255 - m_InputImage[rgb][h][w]
    displayImage()


## 전역 변수
### 윈도우창, 캔버스, 이미지용 화면
window, canvas, paper = None, None, None
### 파일 주소
filename = ""
m_InputImage, m_OutputImage = None, None
m_inH, m_inW, m_outH, m_outW = [0] * 4
RGB, RR, GG, BB = 3, 0, 1, 2

## 메인
window = Tk()
window.title("컬러 영상처리 Ver 0.3")  # 타이틀
window.geometry('500x500')  # 윈도우창 크기
window.resizable(width=False, height=True)  # 사이즈 조정여부
mainMenu = Menu(window)  # 메뉴 창
status = Label(window, text='이미지 정보', bd=1, relief=SUNKEN, anchor=W)  # 상태바
### 메뉴창 선언
window.config(menu=mainMenu)  # 세부설정(메뉴 창)
#### 메뉴창_파일
fileMenu = Menu(mainMenu)
mainMenu.add_cascade(label="파일", menu=fileMenu)
fileMenu.add_command(label="열기", command=openImage)
fileMenu.add_command(label="저장", command=saveImage)
fileMenu.add_separator()
fileMenu.add_command(label="종료", command=None)
#### 메뉴창_화소점처리
pxpointMenu = Menu(mainMenu)
mainMenu.add_cascade(label="화소점필터", menu=pxpointMenu)
pxpointMenu.add_command(label="동일", command=equalImage)
pxpointMenu.add_command(label="반전", command=reverseImage)

#### 캔버스
canvas = Canvas(window, height=500, width=500)  # 캔버스 위젯
paper = PhotoImage(width=500, height=500)  # paper 초기화
canvas.create_image((500 / 2, 500 / 2), image=paper, state='normal')  # 캔버스 중앙에 paper 그리기
##### 리스트는 읽기쓰기 가능, 튜플 읽기만 됨.

canvas.pack()

window.mainloop()
