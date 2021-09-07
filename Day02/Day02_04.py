## GUI
#import tkinter
from tkinter import *
from tkinter import messagebox

## 함수
def btnClick():
    messagebox.showinfo("버튼", "나를 클릭했군요")

def clickWindow1(event):
    messagebox.showinfo("윈도", "윈도 왼쪽 버튼 클릭")

def clickWindow3(event):
    messagebox.showinfo("윈도", "윈도 오른쪽 버튼 클릭")

def clickWindow(event):
    if event.num == 1:
        messagebox.showinfo("윈도", "윈도 왼쪽 버튼 클릭")
    if event.num == 2:
        messagebox.showinfo("윈도", "윈도 휠 버튼 클릭")
    if event.num == 3:
        messagebox.showinfo("윈도", "윈도 오른쪽 버튼 클릭")
    x = event.x
    y = event.y
    label1.configure(text="("+str(x) + ","+str(y)+")")

def btnRClick(event):
    x = event.x
    y = event.y
    label1.configure(text="버튼("+str(x) +","+str(y)+")")

# 전역
window = Tk()
window.title("요기 제목") # 타이틀
window.geometry('400x200') # 윈도우창 크기
window.resizable(width=False, height=True) # 사이즈 조정여부
mainMenu=Menu(window) # 메뉴 창
window.config(menu=mainMenu) # 세부설정(메뉴 창)

fileMenu = Menu(mainMenu)
mainMenu.add_cascade(label = "파일", menu =fileMenu)
fileMenu.add_command(label="열기")
fileMenu.add_separator()
fileMenu.add_command(label="저장")


label1 = Label(window, text="안녕하세요")
label2 = Label(window, text="그래", font=("궁서체", 30), fg="blue")
label3 = Label(window, text = "고맙다", bg = "magenta", width = 20, height = 5)
button1 = Button(window, text ='나를 클릭~', fg ='red', bg ='yellow',command=btnClick)
        # command에는 함수() 가 아닌 함수이름만 나와야함.(콜백함수)

button2 = Button(window, text ='흑백', fg ='white', bg ='black',command=btnClick)
button3 = Button(window, text ='반전', fg ='red', bg ='yellow',command=btnClick)
button4 = Button(window, text ='확대', fg ='black', bg ='white',command=btnClick)

# 메인
# label1.pack(side=LEFT)
# label2.pack()
# label3.pack()
# button1.pack(side=BOTTOM)
label1.place(x=160, y=100)
button1.pack(ipadx=20, ipady=20)
button2.pack(ipadx=20, ipady=20)
button3.pack(ipadx=20, ipady=20)
button4.pack(ipadx=20, ipady=20)

# window.bind("<Button-1>", clickWindow1) # 마우스 왼쪽
# window.bind("<Button-3>", clickWindow3) # 마우스 오른쪽
# window.bind("<Button>", clickWindow)
window.bind("<Button>", btnRClick)
window.mainloop()

