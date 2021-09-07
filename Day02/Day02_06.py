# 그레이 영상처리 Ver 2.2 -->  칼라 영상처리 Ver 2.5 (목) --> 금 정리+ 다음주 예습
from tkinter.colorchooser import *  # 컬러 스케일 사용가능
from tkinter.simpledialog import *
from tkinter.filedialog import *
import os
import math


## 함수
### 공통함수
def malloc2D(h, w):
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    # 지역변수
    memory = [[0 for _ in range(w)] for _ in range(h)]  # 리스트 함축
    # tmpAry = []
    # for _ in range(height):
    #     tmpAry = []
    #     for _ in range(width):
    #         tmpAry.append(0)
    #     memory.append(tmpAry)
    return memory


def malloc2D_double(h, w):
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    # 지역변수
    memory = [[0.0 for _ in range(w)] for _ in range(h)]  # 리스트 함축
    # tmpAry = []
    # for _ in range(height):
    #     tmpAry = []
    #     for _ in range(width):
    #         tmpAry.append(0)
    #     memory.append(tmpAry)
    return memory


def openImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    # 파일 선택하고 크기 계산
    ## 파일 이름으로 파일 찾기
    filename = askopenfilename(parent=window, filetypes=(("RAW파일", "*.raw"), ("모든파일", "*.*")))
    rfp = open(filename, "rb")
    fsize = os.path.getsize(filename)
    m_inH = m_inW = int(math.sqrt(fsize))
    # 메모리 할당
    m_InputImage = malloc2D(m_inH, m_inW)
    # 파일 --> 메모리
    for h in range(m_inH):
        for w in range(m_inW):
            m_InputImage[h][w] = int(ord(rfp.read(1)))
    rfp.close()
    print(m_InputImage[100][100])
    equalImage()


def displayImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    # 캔버스
    ## 캔버스 초기화(이전것 없애기)
    if canvas != None:
        canvas.destroy()
    ## 윈도우 화면 크기 딱 맞추기
    window.geometry(str(m_outH) + "x" + str(m_outW))
    canvas = Canvas(window, height=m_outH, width=m_outW)  # 캔버스 위젯
    paper = PhotoImage(height=m_outH, width=m_outW)  # paper 초기화
    canvas.create_image((m_outH / 2, m_outW / 2), image=paper, state='normal')  # 캔버스 중앙에 paper 그리기
    # 메모리 --> 화면에 찍기
    ## 일반출력(파이썬식, 느림)
    # for i in range(m_outH):
    #     for k in range(m_outW):
    #         # 그레이 스케일 색상담기
    #         r = g = b = m_OutputImage[i][k]
    #         # paper에 16진수로 #RGB값을 넘긴다.
    #         paper.put("#%02x%02x%02x" % (r, g, b), (k, i) )
    ## 빠른출력(C++식, 더블버퍼, 빠름)
    rgbString = ""
    for i in range(m_outH):
        tmpStr = ""  # 한 라인
        for k in range(m_outW):
            r = g = b = m_OutputImage[i][k]
            # 'rgb ' 로 넣기 (마지막 띄어쓰기로 구분처리(구분처리 안하면 못읽음))
            tmpStr += "#%02x%02x%02x " % (r, g, b)
        # '{rgb} ' 로 넣기 (마지막 띄어쓰기로 구분처리(구분처리 안하면 못읽음))
        rgbString += "{" + tmpStr + "} "
    paper.put(rgbString)
    # canvas 출력
    canvas.pack()


### 영상처리함수
#### 동일 이미지
def equalImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    # (중요!) 출력 영상 크기 결정 --> 알고리즘에 따름
    m_outH = m_inH
    m_outW = m_inW
    # 출력 메모리 할당
    m_OutputImage = malloc2D(m_outH, m_outW)
    ## 진짜 영상처리 알고리즘
    for h in range(m_inH):
        for w in range(m_inW):
            m_OutputImage[h][w] = m_InputImage[h][w]
    displayImage()


#### 반전이미지
def reverseImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[반전이미지]")
    for i in range(m_outH):
        for k in range(m_outW):
            m_OutputImage[i][k] = 255 - m_InputImage[i][k]
    displayImage()


#### 127흑백이미지
def bw127Image():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[127흑백이미지]")
    for i in range(m_outH):
        for k in range(m_outW):
            if m_InputImage[i][k] > 127:
                m_OutputImage[i][k] = 255
            else:
                m_OutputImage[i][k] = 0
    displayImage()


#### 평균흑백이미지
def bwAvgImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[평균흑백이미지]")
    avg = 0
    for i in range(m_outH):
        for k in range(m_outW):
            avg = m_InputImage[i][k]
    avg /= len(m_InputImage)
    for i in range(m_outH):
        for k in range(m_outW):
            if (m_InputImage[i][k] > avg):
                m_OutputImage[i][k] = 255
            else:
                m_OutputImage[i][k] = 0
    displayImage()


#### 더하기밝기이미지
def lightPlusImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[더하기밝기이미지]")
    number = 100
    for i in range(m_outH):
        for k in range(m_outW):
            if (m_InputImage[i][k] + 100) > 255:
                m_OutputImage[i][k] = 255
            else:
                m_OutputImage[i][k] = m_InputImage[i][k] + 100
    displayImage()


#### 감마이미지(1.6) *******************오류
def gammaImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[감마이미지(1.6)]")
    gamma = 1.6
    for i in range(m_outH):
        for k in range(m_outW):
            m = m_InputImage[i][k]
            m_OutputImage[i][k] = 255.0 * pow(m / 255.0, gamma)
    displayImage()


#### Cap파라이미지 *******************오류
def paraCapImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[Cap파라이미지]")
    for i in range(m_outH):
        for k in range(m_outW):
            value = 255.0 - 255.0 * math.pow((m_InputImage[i][k] / 128.0 - 1.0), 2);  # 밝은 곳 입체형 (CAP)
            if (value > 255.0):
                value = 255.0;
            elif (value < 0.0):
                value = 0.0;
        m_OutputImage[i][k] = value
    displayImage()


#### 2배확대이미지
def zoomInImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    scale = 2
    m_outH = m_inH * scale
    m_outW = m_inW * scale
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[2배확대이미지]")
    for i in range(m_outH):
        for k in range(m_outW):
            m = m_InputImage[int(i / scale)][int(k / scale)]
            m_OutputImage[i][k] = m
    displayImage()


#### 2배축소이미지
def zoomOutImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    scale = 2
    m_outH = int(m_inH / scale)
    m_outW = int(m_inW / scale)
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[2배축소이미지]")
    for i in range(m_outH):
        for k in range(m_outW):
            m = m_InputImage[int(i * scale)][int(k * scale)]
            m_OutputImage[i][k] = m
    displayImage()


#### 2배양선형확대이미지
def zoomYSInImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    print("[2배양선형확대이미지]")
    scale = 2
    m_outH = int(m_inH * scale)
    m_outW = int(m_inW * scale)
    tempImage = malloc2D(m_inH, m_inW)
    m_OutputImage = malloc2D(m_outH, m_outW)
    for i in range(m_inH):
        for k in range(m_inW):
            tempImage[i][k] = m_InputImage[i][k]

    r_H, r_W, s_H, s_W = [0.0] * 4
    i_W, i_H, v = [0] * 3
    for i in range(m_outH):
        for k in range(m_outW):
            r_H = i / scale;
            r_W = k / scale;

            i_H = int(math.floor(r_H))  # 내림.celi = 올림
            i_W = int(math.floor(r_W))

            s_H = r_H - i_H;
            s_W = r_W - i_W;

            if (i_H < 0 or i_H >= (m_inH - 1) or i_W < 0 or i_W >= (m_inW - 1)):
                m_OutputImage[i][k] = 255
            else:
                C1 = tempImage[i_H][i_W]  # A
                C2 = tempImage[i_H][i_W + 1]  # B
                C3 = tempImage[i_H + 1][i_W + 1]  # C
                C4 = tempImage[i_H + 1][i_W]  # D
                v = int(C1 * (1 - s_H) * (1 - s_W) + C2 * s_W * (1 - s_H) + C3 * s_W * s_H + C4 * (1 - s_W) * s_H)
                m_OutputImage[i][k] = v

    displayImage()


#### 회전(30도)이미지
def roateImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    angle = 30
    tmp_radian = angle % 90 * 3.141592 / 180.0;
    tmp_radian90 = (90 - angle % 90) * 3.141592 / 180.0;
    # 출력 영상의 높이와 폭을 결정 --> 알고리즘에 따름
    m_outH = (int)(m_inH * math.cos(tmp_radian90) + m_inW * math.cos(tmp_radian));
    m_outW = (int)(m_inW * math.cos(tmp_radian) + m_inW * math.cos(tmp_radian90));
    # 출력 영상 메모리 할당
    radian = angle * 3.141592 / 180.0
    m_OutputImage = malloc2D(m_outH, m_outW)
    # 임시 입력 영상 ---> 출력과 크기가 같게 하고, 입력 영상을 중앙에 두기.
    tmpInput = malloc2D(m_outH, m_outW)
    dx = int((m_outH - m_inH) / 2)
    dy = int((m_outW - m_inW) / 2)
    # 임시 입력 영상 ---> 출력과 크기가 같게 하고, 입력 영상을 중앙에 두기.
    for i in range(m_inH):
        for k in range(m_inW):
            tmpInput[i + dx][k + dy] = m_InputImage[i][k]
    # 중앙 위치 구하기
    cx = int(m_outH / 2)
    cy = int(m_outW / 2)
    print("[회전(30도)이미지]")
    for i in range(m_outH):
        for k in range(m_outW):
            oldI = int((math.cos(radian) * (i - cx) + math.sin(radian) * (k - cy)) + cx)
            oldK = int((-math.sin(radian) * (i - cx) + math.cos(radian) * (k - cy)) + cy)
            if (((0 <= oldI) and (oldI < m_outH)) and ((0 <= oldK) and (oldK < m_outW))):
                m_OutputImage[i][k] = tmpInput[oldI][oldK]
    displayImage()


#### 이동이미지(100)
def moveImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    input = 100
    m_outH = m_inH + input
    m_outW = m_inW + input
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[이동이미지(100)]")
    for i in range(m_inH):
        for k in range(m_inW):
            m_OutputImage[i + input][k + input] = m_InputImage[i][k];
    displayImage()


#### 상하미러링
def mirrorUpImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[상하미러링]")
    for i in range(m_outH):
        for k in range(m_outW):
            m_OutputImage[i][k] = m_InputImage[m_inH - i - 1][k]
    displayImage()


#### 좌우미러링
def mirrorLRImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[좌우미러링]")
    for i in range(m_outH):
        for k in range(m_outW):
            m_OutputImage[i][k] = m_InputImage[i][m_inW - k - 1];
    displayImage()


#### 엠보싱이미지
def embosImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)

    print("[엠보싱이미지]")
    # 마스크
    mask = [[-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]]
    # 임시 입/출력 메모리 준비
    tmpInput = malloc2D_double(m_inH + 2, m_inW + 2)
    tmpOutput = malloc2D_double(m_outH, m_outW)
    for i in range(m_inH):
        for k in range(m_inW):
            tmpInput[i + 1][k + 1] = m_InputImage[i][k]
    # 마스크 회선 연산
    for i in range(m_inH):
        for k in range(m_inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInput[i + m][k + n] * mask[m][n]
            tmpOutput[i][k] = S
    # 다듬기
    for i in range(m_outH):
        for k in range(m_outW):
            tmpOutput[i][k] += 127.0
    ## 오버플로 확인절차 및 출력
    for i in range(m_outH):
        for k in range(m_outW):
            v = tmpOutput[i][k]
            if (v > 255.0): v = 255.0
            if (v < 0.0): v = 0.0
            m_OutputImage[i][k] = int(v)
    displayImage()


#### 평균블러이미지(3x3)
def blurAvgImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)

    print("[평균블러이미지(3x3)]")
    # 마스크 ( 입력값에 맞게 블러를 만들어준다 )
    scale = 3
    if scale % 2 == 0:
        scale += 1
    mask = malloc2D_double(scale, scale)
    for i in range(scale):
        for k in range(scale):
            mask[i][k] = 1.0 / math.pow(scale, 2)
    # 임시 입/출력 메모리 준비
    tmpInput = malloc2D_double(m_inH + 2, m_inW + 2)
    tmpOutput = malloc2D_double(m_outH, m_outW)
    for i in range(m_inH):
        for k in range(m_inW):
            tmpInput[i + 1][k + 1] = m_InputImage[i][k]
    # 마스크 회선 연산 
    for i in range(m_inH):
        for k in range(m_inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInput[i + m][k + n] * mask[m][n]
            tmpOutput[i][k] = S
    ## 오버플로 확인절차 및 출력
    for i in range(m_outH):
        for k in range(m_outW):
            v = tmpOutput[i][k]
            if (v > 255.0): v = 255.0
            if (v < 0.0): v = 0.0
            m_OutputImage[i][k] = int(v)
    displayImage()


#### 가우스블러(3x3)처리
def gausBlrImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)

    print("[가우스블러처리]")
    # 마스크
    mask = [[1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0],
            [1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0],
            [1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0]]
    # 임시 입/출력 메모리 준비
    tmpInput = malloc2D_double(m_inH + 2, m_inW + 2)
    tmpOutput = malloc2D_double(m_outH, m_outW)
    for i in range(m_inH):
        for k in range(m_inW):
            tmpInput[i + 1][k + 1] = m_InputImage[i][k]
    # 마스크 회선 연산
    for i in range(m_inH):
        for k in range(m_inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInput[i + m][k + n] * mask[m][n]
            tmpOutput[i][k] = S
    ## 오버플로 확인절차 및 출력
    for i in range(m_outH):
        for k in range(m_outW):
            v = tmpOutput[i][k]
            if (v > 255.0): v = 255.0
            if (v < 0.0): v = 0.0
            m_OutputImage[i][k] = int(v)
    displayImage()


#### 고주파패스필터처리
def hpfImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)

    print("[고주파패스필터처리]")
    # 마스크
    mask = [[0., -1., 0],
            [-1., 5., -1.],
            [0, -1., 0.]]
    # 임시 입/출력 메모리 준비
    tmpInput = malloc2D_double(m_inH + 2, m_inW + 2)
    tmpOutput = malloc2D_double(m_outH, m_outW)
    for i in range(m_inH):
        for k in range(m_inW):
            tmpInput[i + 1][k + 1] = m_InputImage[i][k]
    # 마스크 회선 연산
    for i in range(m_inH):
        for k in range(m_inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInput[i + m][k + n] * mask[m][n]
            tmpOutput[i][k] = S
    ## 오버플로 확인절차 및 출력
    for i in range(m_outH):
        for k in range(m_outW):
            v = tmpOutput[i][k]
            if (v > 255.0): v = 255.0
            if (v < 0.0): v = 0.0
            m_OutputImage[i][k] = int(v)
    displayImage()


#### 고주파필터처리
def lpfImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)

    print("[고주파필터처리]")
    # 마스크
    mask = [[-1. / 9., 1 / 9., -1. / 9.],
            [-1. / 9., 8. / 9., -1. / 8.],
            [-1. / 9., 1 / 9., -1. / 9.]]
    # 임시 입/출력 메모리 준비
    tmpInput = malloc2D_double(m_inH + 2, m_inW + 2)
    tmpOutput = malloc2D_double(m_outH, m_outW)
    for i in range(m_inH):
        for k in range(m_inW):
            tmpInput[i + 1][k + 1] = m_InputImage[i][k]
    # 마스크 회선 연산
    for i in range(m_inH):
        for k in range(m_inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInput[i + m][k + n] * mask[m][n]
            tmpOutput[i][k] = S
    ## 오버플로 확인절차 및 출력
    for i in range(m_outH):
        for k in range(m_outW):
            v = tmpOutput[i][k]
            if (v > 255.0): v = 255.0
            if (v < 0.0): v = 0.0
            m_OutputImage[i][k] = int(v)
    displayImage()


#### 유사연산자처리
def calcUsaImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)

    print("[유사연산자처리]")

    # 임시 입/출력 메모리 준비
    tmpInput = malloc2D_double(m_inH + 2, m_inW + 2)
    tmpOutput = malloc2D_double(m_outH, m_outW)

    for i in range(m_inH):
        for k in range(m_inW):
            tmpInput[i + 1][k + 1] = m_InputImage[i][k]

    ## 유사연산 알고리즘
    for i in range(m_inH):
        for k in range(m_inW):
            max = 0.0
            for m in range(3):
                for n in range(3):
                    if abs(tmpInput[i + 1][k + 1] - tmpInput[i + m][k + n] >= max):
                        # 블록의 가운대값 - 블록의 주변 픽셀값의 절대값 중에서
                        # 최대값을 찾는다.
                        max = abs(tmpInput[i + 1][k + 1] - tmpInput[i + m][k + n])
            tmpOutput[i][k] = max

    ## 오버플로 확인절차 및 출력
    for i in range(m_outH):
        for k in range(m_outW):
            v = tmpOutput[i][k]
            if (v > 255.0): v = 255.0
            if (v < 0.0): v = 0.0
            m_OutputImage[i][k] = int(v)
    displayImage()


#### 차연산자
def calcMnsImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)

    print("[고주파필터처리]")
    # 임시 입/출력 메모리 준비
    tmpInput = malloc2D_double(m_inH + 2, m_inW + 2)
    tmpOutput = malloc2D_double(m_outH, m_outW)
    for i in range(m_inH):
        for k in range(m_inW):
            tmpInput[i + 1][k + 1] = m_InputImage[i][k]
    mask = [0 for _ in range(4)]
    for i in range(m_inH):
        for k in range(m_inW):
            max = 0.0
            mask[0] = abs(tmpInput[i][k] - tmpInput[i + 2][k + 2])
            mask[1] = abs(tmpInput[i][k + 1] - tmpInput[i + 2][k + 1])
            mask[2] = abs(tmpInput[i][k + 2] - tmpInput[i + 2][k])
            mask[3] = abs(tmpInput[i + 1][k] - tmpInput[i + 1][k + 2])
            for m in range(4):
                if (mask[m] >= max): max = mask[m]
            tmpOutput[i][k] = max
    ## 오버플로 확인절차 및 출력
    for i in range(m_outH):
        for k in range(m_outW):
            v = tmpOutput[i][k]
            if (v > 255.0): v = 255.0
            if (v < 0.0): v = 0.0
            m_OutputImage[i][k] = int(v)
    displayImage()


#### 로버츠엣지처리
def robertsImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)

    print("[로버츠엣지처리]")
    # 마스크
    maskV = [[-1, 0, 0],
             [0, 1, 0],
             [0, 0, 0]]
    maskH = [[0, 0, -1],
             [0, 1, 0],
             [0, 0, 0]]
    # 임시 입/출력 메모리 준비
    tmpInput = malloc2D_double(m_inH + 2, m_inW + 2)
    tmpOutput = malloc2D_double(m_outH, m_outW)
    tmpOutput2 = malloc2D_double(m_outH, m_outW)
    for i in range(m_inH):
        for k in range(m_inW):
            tmpInput[i + 1][k + 1] = m_InputImage[i][k]
    # 마스크 회선 연산
    for i in range(m_inH):
        for k in range(m_inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInput[i + m][k + n] * maskH[m][n]
            tmpOutput[i][k] = S
    for i in range(m_inH):
        for k in range(m_inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInput[i + m][k + n] * maskV[m][n]
            tmpOutput2[i][k] = S
    ## 오버플로 확인절차 및 출력
    for i in range(m_outH):
        for k in range(m_outW):
            v1 = tmpOutput[i][k]
            v2 = tmpOutput2[i][k]
            v = v1 + v2
            if (v > 255.0): v = 255.0
            if (v < 0.0): v = 0.0
            m_OutputImage[i][k] = int(v)
    displayImage()


#### 프리윗엣지처리
def prwImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)

    print("[로버츠엣지처리]")
    # 마스크
    maskV = [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]
    maskH = [[1, 1, 1],
             [0, 0, 0],
             [-1, -1, -1]]
    # 임시 입/출력 메모리 준비
    tmpInput = malloc2D_double(m_inH + 2, m_inW + 2)
    tmpOutput = malloc2D_double(m_outH, m_outW)
    tmpOutput2 = malloc2D_double(m_outH, m_outW)
    for i in range(m_inH):
        for k in range(m_inW):
            tmpInput[i + 1][k + 1] = m_InputImage[i][k]
    # 마스크 회선 연산
    for i in range(m_inH):
        for k in range(m_inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInput[i + m][k + n] * maskH[m][n]
            tmpOutput[i][k] = S
    for i in range(m_inH):
        for k in range(m_inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInput[i + m][k + n] * maskV[m][n]
            tmpOutput2[i][k] = S
    ## 오버플로 확인절차 및 출력
    for i in range(m_outH):
        for k in range(m_outW):
            v1 = tmpOutput[i][k]
            v2 = tmpOutput2[i][k]
            v = v1 + v2
            if (v > 255.0): v = 255.0
            if (v < 0.0): v = 0.0
            m_OutputImage[i][k] = int(v)
    displayImage()


#### 소벨엣지처리
def sblImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)

    print("[소벨엣지처리]")
    # 마스크
    maskV = [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]]
    maskH = [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
    # 임시 입/출력 메모리 준비
    tmpInput = malloc2D_double(m_inH + 2, m_inW + 2)
    tmpOutput = malloc2D_double(m_outH, m_outW)
    tmpOutput2 = malloc2D_double(m_outH, m_outW)
    for i in range(m_inH):
        for k in range(m_inW):
            tmpInput[i + 1][k + 1] = m_InputImage[i][k]
    # 마스크 회선 연산
    for i in range(m_inH):
        for k in range(m_inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInput[i + m][k + n] * maskH[m][n]
            tmpOutput[i][k] = S
    for i in range(m_inH):
        for k in range(m_inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInput[i + m][k + n] * maskV[m][n]
            tmpOutput2[i][k] = S
    ## 오버플로 확인절차 및 출력
    for i in range(m_outH):
        for k in range(m_outW):
            v1 = tmpOutput[i][k]
            v2 = tmpOutput2[i][k]
            v = v1 + v2
            if (v > 255.0): v = 255.0
            if (v < 0.0): v = 0.0
            m_OutputImage[i][k] = int(v)
    displayImage()


#### 라플라필터처리
def laplaImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)

    print("[라플라필터처리]")
    # 마스크
    mask = [[-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]]
    # 임시 입/출력 메모리 준비
    tmpInput = malloc2D_double(m_inH + 2, m_inW + 2)
    tmpOutput = malloc2D_double(m_outH, m_outW)
    for i in range(m_inH):
        for k in range(m_inW):
            tmpInput[i + 1][k + 1] = m_InputImage[i][k]
    # 마스크 회선 연산
    for i in range(m_inH):
        for k in range(m_inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInput[i + m][k + n] * mask[m][n]
            tmpOutput[i][k] = S
    ## 오버플로 확인절차 및 출력
    for i in range(m_outH):
        for k in range(m_outW):
            v = tmpOutput[i][k]
            if (v > 255.0): v = 255.0
            if (v < 0.0): v = 0.0
            m_OutputImage[i][k] = int(v)
    displayImage()

#### LoG필터처리
def logFilterImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)

    print("[LoG필터처리]")
    # 마스크
    mask = [[0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]];
    # 임시 입/출력 메모리 준비
    tmpInput = malloc2D_double(m_inH + 4, m_inW + 4)
    tmpOutput = malloc2D_double(m_outH, m_outW)
    for i in range(m_inH):
        for k in range(m_inW):
            tmpInput[i + 2][k + 2] = m_InputImage[i][k]
    # 마스크 회선 연산
    for i in range(m_inH):
        for k in range(m_inW):
            S = 0.0
            for m in range(5):
                for n in range(5):
                    S += tmpInput[i + m][k + n] * mask[m][n]
            tmpOutput[i][k] = S
    ## 오버플로 확인절차 및 출력
    for i in range(m_outH):
        for k in range(m_outW):
            v = tmpOutput[i][k]
            if (v > 255.0): v = 255.0
            if (v < 0.0): v = 0.0
            m_OutputImage[i][k] = int(v)
    displayImage()

#### DoG필터처리 ************ 오류
def dogFilterImage():
    global window, canvas, paper, m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)

    print("[DoG필터처리]")
    # 마스크
    mask = [[0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]]
    # 임시 입/출력 메모리 준비
    tmpInput = malloc2D_double(m_inH + 8, m_inW + 8)
    tmpOutput = malloc2D_double(m_outH, m_outW)
    for i in range(m_inH):
        for k in range(m_inW):
            tmpInput[i + 4][k + 4] = m_InputImage[i][k]
    # 마스크 회선 연산
    for i in range(m_inH):
        for k in range(m_inW):
            S = 0.0
            for m in range(9):
                for n in range(9):
                    S += tmpInput[i + m][k + n] * mask[m][n]
            tmpOutput[i][k] = S
    ## 오버플로 확인절차 및 출력
    for i in range(m_outH):
        for k in range(m_outW):
            v = tmpOutput[i][k]
            if (v > 255.0): v = 255.0
            if (v < 0.0): v = 0.0
            m_OutputImage[i][k] = int(v)
    displayImage()


## 전역 변수
### 윈도우창, 캔버스, 이미지용 화면
window, canvas, paper = None, None, None
### 파일 주소
filename = ""
m_InputImage, m_OutputImage = None, None
m_inH, m_inW, m_outH, m_outW = [0] * 4

## 메인
window = Tk()
window.title("그레이 영상처리 Ver 0.2")  # 타이틀
window.geometry('500x500')  # 윈도우창 크기
window.resizable(width=False, height=True)  # 사이즈 조정여부
mainMenu = Menu(window)  # 메뉴 창
### 메뉴창 선언
window.config(menu=mainMenu)  # 세부설정(메뉴 창)
#### 메뉴창_파일
fileMenu = Menu(mainMenu)
mainMenu.add_cascade(label="파일", menu=fileMenu)
fileMenu.add_command(label="열기", command=openImage)
fileMenu.add_command(label="저장", command=None)
fileMenu.add_separator()
fileMenu.add_command(label="종료", command=None)
#### 메뉴창_화소점처리
pxpointMenu = Menu(mainMenu)
mainMenu.add_cascade(label="화소점필터", menu=pxpointMenu)
pxpointMenu.add_command(label="동일", command=equalImage)
pxpointMenu.add_command(label="반전", command=reverseImage)
pxpointMenu.add_command(label="127흑백", command=bw127Image)
pxpointMenu.add_command(label="평균흑백", command=bwAvgImage)
pxpointMenu.add_command(label="더하기밝기", command=lightPlusImage)
pxpointMenu.add_command(label="감마(1.6)", command=gammaImage)
pxpointMenu.add_command(label="파라볼라(cap)", command=paraCapImage)
#### 메뉴창_기하학처리
ghhMenu = Menu(mainMenu)
mainMenu.add_cascade(label="기하학처리", menu=ghhMenu)
ghhMenu.add_command(label="이동이미지", command=moveImage)
ghhMenu.add_command(label="상하미러링", command=mirrorUpImage)
ghhMenu.add_command(label="좌우미러링", command=mirrorLRImage)
ghhMenu.add_command(label="2배확대", command=zoomInImage)
ghhMenu.add_command(label="2배축소", command=zoomOutImage)
ghhMenu.add_command(label="2배양선형확대", command=zoomYSInImage)
ghhMenu.add_command(label="중앙회전(30도)", command=roateImage)
#### 메뉴창_화소영역처리
pxAreaMenu = Menu(mainMenu)
mainMenu.add_cascade(label="화소영역필터", menu=pxAreaMenu)
pxAreaMenu.add_command(label="엠보싱", command=embosImage)
pxAreaMenu.add_command(label="평균블러(3x3)", command=blurAvgImage)
pxAreaMenu.add_command(label="가우스블러(3x3)", command=gausBlrImage)
pxAreaMenu.add_command(label="고주파패스필터", command=hpfImage)
pxAreaMenu.add_command(label="고주파필터", command=lpfImage)
#### 메뉴창_화소영역처리(엣지처리)
pxAreaEdgeMenu = Menu(mainMenu)
mainMenu.add_cascade(label="엣지필터", menu=pxAreaEdgeMenu)
pxAreaEdgeMenu.add_command(label="유사연산", command=calcUsaImage)
pxAreaEdgeMenu.add_command(label="차연산", command=calcMnsImage)
pxAreaEdgeMenu.add_command(label="로버츠", command=robertsImage)
pxAreaEdgeMenu.add_command(label="프리윗", command=prwImage)
pxAreaEdgeMenu.add_command(label="소벨", command=sblImage)
pxAreaEdgeMenu.add_command(label="라플라", command=laplaImage)
pxAreaEdgeMenu.add_command(label="LoG", command=logFilterImage)
pxAreaEdgeMenu.add_command(label="DoG", command=dogFilterImage)
#### 메뉴창_히스토그램처리
histogramMenu = Menu(mainMenu)
mainMenu.add_cascade(label="히스토그램", menu=histogramMenu)
histogramMenu.add_command(label="스트레칭", command=moveImage)
histogramMenu.add_command(label="평활화", command=mirrorUpImage)
histogramMenu.add_separator()
histogramMenu.add_command(label="출력", command=mirrorLRImage)

#### 캔버스
canvas = Canvas(window, height=500, width=500)  # 캔버스 위젯
paper = PhotoImage(width=500, height=500)  # paper 초기화
canvas.create_image((500 / 2, 500 / 2), image=paper, state='normal')  # 캔버스 중앙에 paper 그리기
##### 리스트는 읽기쓰기 가능, 튜플 읽기만 됨.

canvas.pack()

window.mainloop()
