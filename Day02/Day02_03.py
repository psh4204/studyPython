import random
## 공통 함수
def malloc2D(height, width):
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    # 지역변수
    memory = []
    tmpAry = []
    for _ in range(height):
        tmpAry = []
        for _ in range(width):
            tmpAry.append(0)
        memory.append(tmpAry)
    return memory

def openImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    # 파일 선택하고 크기 계산
    filename = "C:\\images\\meRAW\\me512.raw"
    rfp = open(filename, "rb")
    m_inH = m_inW = 512
    # 메모리 할당
    m_InputImage = malloc2D(m_inH,m_inW)
    # 파일 --> 메모리
    for h in range(m_inH):
        for w in range(m_inW):
            m_InputImage[h][w] = random.randint(0, 255)
    rfp.close()
    equalImage()


def displayImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    for h in range(10):
        for w in range(10):
            print("%3d " % m_InputImage[h+100][w+100], end="")
        print()
    print()
    for h in range(10):
        for w in range(10):
            print("%3d " % m_OutputImage[h+100][w+100], end="")
        print()
    print()

## 영상처리 함수
### 동일 이미지
def equalImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
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

### 반전 이미지
def reverseImage():
    global m_InputImage, m_OutputImage, m_inH, m_inW, m_outH, m_outW
    # (중요!) 출력 영상 크기 결정 --> 알고리즘에 따름
    m_outH = m_inH
    m_outW = m_inW
    # 출력 메모리 할당
    m_OutputImage = malloc2D(m_outH, m_outW)
    ## 진짜 영상처리 알고리즘
    for h in range(m_inH):
        for w in range(m_inW):
            m_OutputImage[h][w] = 255 - m_InputImage[h][w]
    displayImage()


## 변수
m_InputImage, m_OutputImage = None, None
m_inH, m_inW, m_outH, m_outW = [0] * 4

## 메인
# 파일 선택 하고 열기
openImage() # 파이썬함수는 소문자시작, 동사로.
reverseImage() # 파이썬함수는 소문자시작, 동사로.