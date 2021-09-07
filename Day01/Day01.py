import math

# 두 숫자를 입력받아서, 더하기 / 빼기 .. 출력하기.
## 함수 선언부


## 전역 변수부
num1, nu2 = 0, 0
res = 0
## 메인 코드부
num1 = int(input("숫자1 -->"))
num2 = int(input("숫자2 -->"))

# 퀴즈1. 빼기, 곱하기, 나누기, 나머지, 몫, 제곱 계산하기...
res = num1 + num2
print(num1, "+", num2, "=", res)
res = num1 - num2
print(num1, "-", num2, "=", res)
res = num1 * num2
print(num1, "에", num2, "나눈 후 나머지", "=", res)
res = num1 / num2
print(num1, "/", num2, "=", res)
res = num1 // num2
print(num1, "//", num2, "=", res)
res = num1 ** num2
print(num1, "의", num2, "제곱", "=", res)
res = math.pow(num1, num2)
print(num1, "의", num2, "제곱", "=", res)
res = math.sqrt(num1)
print("루트", num1, "=", int(res))
