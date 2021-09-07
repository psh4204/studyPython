## 함수

## 변수

## 메인
# 퀴즈2 : 77 부터 2345 까지 홀수의 합
# for i in range(1,101, 1): # for(int i = 1; i<101; i++)
#     hap += i
hap = 0
for i in range(77, 2345 + 1, 1):
    if i % 2 == 1:
        hap += i
print("퀴즈2 : 77 부터 2345 까지 홀수의 합 : ", hap)
# 퀴즈3 : 777 부터 23456 까지 519의 배수의 합
hap = 0
for i in range(777, 23456 + 1, 1):
    if i % 519 == 0:
        hap += i
print("퀴즈3 : 777 부터 23456 까지 519의 배수의 합 : ", hap)
# 퀴즈4 : 2부터 10000까지 소수의 합
hap = 2
for i in range(3, 10000 + 1, 1):
    sosuYN = True
    for k in range(2, int(i/2) + 1, 1):
        if i % k == 0:
            sosuYN = False
            break
    if sosuYN:
        hap += i
print("퀴즈4 : 2부터 10000까지 소수의 합 : ", hap)