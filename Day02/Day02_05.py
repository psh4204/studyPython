from tkinter import *
from tkinter.colorchooser import * #컬러 스케일 사용가능
from tkinter.simpledialog import *

## 함수
def getColor():
    global window, canvas, pColor, pWidth
    #색상 묻기
    clr = askcolor()
    pColor = clr[1]

def getWidth():
    global window, canvas, pColor, pWidth
    # 정수 묻기
    pWidth = askinteger("선두께", "정수를 입력", minvalue=0, maxvalue=10)

def mouseClick(event):
    global window, canvas, pColor, pWidth, x1, y1, x2, y2

    x1 = event.x
    y1 = event.y


def mouseRelease(event):
    global window, canvas, pColor, pWidth, x1, y1, x2, y2
    # 좌 클릭시 선 그리기
    x2 = event.x
    y2 = event.y
    if event.num == 1:
        canvas.create_line(x1, y1, x2, y2, width=pWidth, fill=pColor)
    # 휠 클릭시 네모 그리기
    if event.num == 2:
        canvas.create_rectangle(x1, y1, x2, y2, width=pWidth, fill=pColor)
    # 우 클릭시 원 그리기
    ## 심화 : 원 중심을 기점으로 그림그려보기
    if event.num == 3:
        tmp = 0
        canvas.create_oval(2*x1-x2,2*y1-y2, x2, y2, width=pWidth, fill=pColor)

## 변수
window, canvas = None, None
pColor, pWidth = "black", 0
x1, y1, x2, y2 = [-1] * 4 # 초기 좌표값 (안보이게 -1좌표에)

## 메인
window = Tk()
mainMenu=Menu(window)
window.config(menu=mainMenu)

### 캔버스
canvas = Canvas(window, height=400, width=400) # 캔버스 위젯
canvas.bind("<Button>", mouseClick)
canvas.bind("<ButtonRelease>", mouseRelease)

# canvas.create_line(0,0,200,200,width=5,fill="red") # 선그리기
# canvas.create_rectangle(100,100,200,200,width=5, outline="blue") # 네모그리기
# canvas.create_oval(100,100,200,200,width=5) # 원 그리기
# canvas.create_arc(100,300,200,200,extent=30) # 호 그리기
# canvas.create_arc(20,300,200,200,extent=90,style=ARC) # 호 그리기

### 메뉴창_파일
fileMenu = Menu(mainMenu)
mainMenu.add_cascade(label="파일", menu =fileMenu)
fileMenu.add_command(label="열기",command=None)
fileMenu.add_separator()
fileMenu.add_command(label="저장",command=None)

### 메뉴팡_설정
setupMenu = Menu(mainMenu)
mainMenu.add_cascade(label="설정", menu =setupMenu)
setupMenu.add_command(label="선색상", command=getColor)
setupMenu.add_separator()
setupMenu.add_command(label="선 두께",command=getWidth)

canvas.pack()
window.mainloop()