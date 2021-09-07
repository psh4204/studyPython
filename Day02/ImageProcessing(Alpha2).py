import random
import math
## 함수
### 공통함수
def malloc2D(h, w):
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    # 지역변수
    memory = []
    tmpArr = []
    for _ in range(h):
        tmpArr = []
        for _ in range(w):
            tmpArr.append(0)
        memory.append(tmpArr)
    return memory

def openImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW

    filename = "C:\\images\\meRAW\\me512.raw"
    rfp = open(filename, "rb")
    m_inH = m_inW = 512

    m_InputImage = malloc2D(m_inH, m_inW)

    for i in range(m_inH):
        for k in range(m_inW):
            m_InputImage[i][k] = random.randint(0, 255)
            rfp.close()
    equalImage()
            

def displayImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    for i in range(10):
        for k in range(10):
            print("%3d " % m_OutputImage[i+100][k+100], end="")
        print()
    print()


### 영상처리함수
#### 동일이미지
def equalImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[동일이미지]")
    for i in range(m_outH):
        for k in range(m_outW):
            m_OutputImage[i][k] = m_InputImage[i][k]
    displayImage()

#### 반전이미지
def reverseImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
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
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[127흑백이미지]")
    for i in range(m_outH):
        for k in range(m_outW):
            if m_InputImage[i][k] > 127: m_OutputImage[i][k] = 255
            else: m_OutputImage[i][k] = 0
    displayImage()

#### 평균흑백이미지
def bwAvgImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
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
            else: m_OutputImage[i][k] = 0
    displayImage()

#### 더하기밝기이미지
def lightPlusImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[더하기밝기이미지]")
    number = 100
    for i in range(m_outH):
        for k in range(m_outW):
            if (m_InputImage[i][k]+100)>255: m_OutputImage[i][k] = 255
            else : m_OutputImage[i][k] = m_InputImage[i][k] + 100
    displayImage()

#### 감마이미지(1.6)
def gammaImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
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

#### Cap파라이미지
def paraCapImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    m_outH = m_inH
    m_outW = m_inW
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[Cap파라이미지]")
    for i in range(m_outH):
        for k in range(m_outW):
            value = 255.0 - 255.0 * math.pow((m_InputImage[i][k] / 128.0 - 1.0), 2); #밝은 곳 입체형 (CAP)
            if (value > 255.0): value = 255.0;
            elif (value < 0.0): value = 0.0;
        m_OutputImage[i][k] = value
    displayImage()

#### 2배확대이미지
def zoomInImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    scale = 2
    m_outH = m_inH * scale
    m_outW = m_inW * scale
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[2배확대이미지]")
    for i in range(m_outH):
        for k in range(m_outW):
            m = m_InputImage[int(i/scale)][int(k/scale)]
            m_OutputImage[i][k] = m
    displayImage()
#### 2배축소이미지
def zoomOutImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    scale = 2
    m_outH = int(m_inH / scale)
    m_outW = int(m_inW / scale)
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[2배축소이미지]")
    for i in range(m_outH):
        for k in range(m_outW):
            m = m_InputImage[int(i/scale)][int(k/scale)]
            m_OutputImage[i][k] = m
    displayImage()

#### 2배양선형확대이미지
def zoomYSInImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
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
    i_W, i_H, v = [0]*3
    for i in range(m_outH):
        for k in range(m_outW):
            r_H = i / scale;
            r_W = k / scale;

            i_H = int(math.floor(r_H)) # 내림.celi = 올림
            i_W = int(math.floor(r_W))

            s_H = r_H - i_H;
            s_W = r_W - i_W;

            if (i_H < 0 or i_H >= (m_inH - 1) or i_W < 0 or i_W >= (m_inW - 1)):
                m_OutputImage[i][k] = 255
            else:
                C1 = tempImage[i_H][i_W] # A
                C2 = tempImage[i_H][i_W + 1] # B
                C3 = tempImage[i_H + 1][i_W + 1] # C
                C4 = tempImage[i_H + 1][i_W] # D
                v = int(C1 * (1 - s_H) * (1 - s_W) + C2 * s_W * (1 - s_H) + C3 * s_W * s_H + C4 * (1 - s_W) * s_H)
                m_OutputImage[i][k] = v

    displayImage()

#### 2배축소이미지
def zoomOutImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    scale = 2
    m_outH = int(m_inH / scale)
    m_outW = int(m_inW / scale)
    m_OutputImage = malloc2D(m_outH, m_outW)
    print("[2배축소이미지]")
    for i in range(m_outH):
        for k in range(m_outW):
            m = m_InputImage[int(i/scale)][int(k/scale)]
            m_OutputImage[i][k] = m
    displayImage()

## 전역변수
m_InputImage, m_OutputImage = None, None
m_inH, m_inW, m_outH, m_outW = [0] * 4

## Main
openImage()
reverseImage()
bw127Image()
bwAvgImage()
lightPlusImage()
gammaImage()
paraCapImage()
zoomInImage()
zoomOutImage()
zoomYSInImage()
zoomOutImage()