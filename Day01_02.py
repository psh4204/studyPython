import math

## 함수 선언부
def addFunc1(n1, n2):
    global gNum, num1, num2 # 전역변수 선언

    # pass # 넘기기
    hap = n1+n2
    gNum = 1234
    return hap

## 전역 변수부
num1, num2 = 0, 0; res = 0
gNum = 0

## 메인 코드부

num1 = int(input("숫자1 -->")); num2 = int(input("숫자2 -->"))

res = addFunc1(num1, num2)
print(num1, '+', num2, '=', res )
print(gNum)