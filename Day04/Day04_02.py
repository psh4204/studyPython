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


def malloc3D_double(h, w, init=0.0):  # *** 수정바람
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
    # 지역변수
    memory = [[[init for _ in range(w)] for _ in range(h)] for _ in range(RGB)]  # 리스트 함축
    return memory


def openImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
    global cvInPhoto, cvOutPhoto
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


### <----- 마우스 클릭 관련  ----> ###
def mouseClick_reverseImage():
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global sx, sy, ex, ey
    # 마우스 이벤트 받을 준비 시키기
    canvas.bind("<Button-1>", leftClickMouse)
    canvas.bind("<ButtonRelease-1>", leftDropMouse_reverseImage)
    canvas.bind("<B1-Motion>", leftMoveMouse) # 움직임대로 상자 그려짐


def leftClickMouse(event):
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global sx, sy, ex, ey
    sx = event.x
    sy = event.y

# 클릭마우스의 움직임 대로 상자 그려짐
def leftMoveMouse(event):
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global sx, sy, ex, ey, boxLine
    ex = event.x
    ey = event.y
    # 이전 박스 지우기
    if boxLine == None:
        pass
    else:
        canvas.delete(boxLine)

    boxLine = canvas.create_rectangle(sx, sy, ex, ey, fill=None)


def leftDropMouse_reverseImage(event):
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global sx, sy, ex, ey
    ex = event.x
    ey = event.y
    # 알고리즘에 맞게 드래그박스 위치 지정
    if sx > ex:
        sx, ex = ex, sx
    if sy > ey:
        sy, ey = ey, sy
    reverseImage_Click()
    canvas.unbind("<Button-1>")
    canvas.unbind("<ButtonRelease-1>")


def reverseImage_Click():
    global window, canvas, paper, filename, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    global sx, sy, ex, ey, boxLine
    # (중요!) 출력 영상 크기 결정 --> 알고리즘에 따름
    m_outH = m_inH
    m_outW = m_inW
    # 출력 메모리 할당
    m_OutputImage = malloc3D(m_outH, m_outW)
    ## 진짜 영상처리 알고리즘
    for rgb in range(RGB):
        for i in range(m_inH):
            for k in range(m_inW):
                if (sx <= k <= ex) and (sy <= i <= ey):
                    m_OutputImage[rgb][i][k] = 255 - m_InputImage[rgb][i][k]
                else:
                    m_OutputImage[rgb][i][k] = m_InputImage[rgb][i][k]
    displayImage()

### OpenCV용 함수
#### 결과를 화면출력으로 보내기
def cv2OutImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
    global cvInPhoto, cvOutPhoto
    # 크기 알아내기
    m_outH, m_outW = cvInPhoto.shape[:2]  # [0:2]0~1까지
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


#### 회색처리_OpenCV
def grayImage_cv():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
    global cvInPhoto, cvOutPhoto
    ## OpenCV용 알고리즘 ##
    cvOutPhoto = cv2.cvtColor(cvInPhoto, cv2.COLOR_RGB2GRAY)
    ## 화면출력
    cv2OutImage()
    displayImage()


#### 엠보싱처리_OpenCV
def embossImage_cv():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
    global cvInPhoto, cvOutPhoto
    ## OpenCV용 알고리즘 ##
    mask = np.zeros((3, 3), np.float32)
    mask[0][0] = -1.0
    mask[2][2] = 1.0
    cvOutPhoto = cv2.filter2D(cvInPhoto, -1, mask)
    cvOutPhoto += 127  # 넘파이배열은 이렇게 간단하게 전부 계산처리 가능
    ## 화면출력
    cv2OutImage()
    displayImage()


#### 카툰이미지처리_OpenCV
def cartoonImage_cv():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW, RGB, RR, GG, BB
    global cvInPhoto, cvOutPhoto
    ## OpenCV용 알고리즘 ##
    cvOutPhoto = cv2.cvtColor(cvInPhoto, cv2.COLOR_RGB2GRAY)
    cvtOutPhoto = cv2.medianBlur(cvOutPhoto, 7)
    edges = cv2.Laplacian(cvOutPhoto, cv2.CV_8U, ksize=5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    cvOutPhoto = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    ## 화면출력
    cv2OutImage()
    displayImage()


## 전역 변수
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
pxpointMenu.add_command(label="지정 반전", command=mouseClick_reverseImage)

#### 메뉴창_OpenCV용 메뉴
openCVMenu = Menu(mainMenu)
mainMenu.add_cascade(label="OpenCV", menu=openCVMenu)
openCVMenu.add_command(label="회색", command=grayImage_cv)
openCVMenu.add_command(label="엠보스", command=embossImage_cv)
openCVMenu.add_command(label="카툰", command=cartoonImage_cv)

#### 캔버스
canvas = Canvas(window, height=500, width=500)  # 캔버스 위젯
paper = PhotoImage(width=500, height=500)  # paper 초기화
canvas.create_image((500 / 2, 500 / 2), image=paper, state='normal')  # 캔버스 중앙에 paper 그리기
##### 리스트는 읽기쓰기 가능, 튜플 읽기만 됨.

canvas.pack()

window.mainloop()
