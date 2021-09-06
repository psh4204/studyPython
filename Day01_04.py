# ## 함수
#
# ## 전역
# ary1 = [0,0,0,0,0] # 리스트 ( = 배열 )
# ary2 = [ ]
# ## 메인
# ary1[0] = 10
# ary1[1] = 20
#
# print(ary1[0], ary1[1])
#
# for _ in range(100): # for i in range(0, 100, 1):
#     ary2.append(0) # 리스트 추가
#
# print(ary2)

# 퀴즈3 : 777 부터 23455까지 519의 배수의 합 ... 먼저 배열에 저장한 후 배열을 합치기
numAry = []
hap = 0

for num in range(777, 23456 + 1, 1):
    numAry.append(num)
for num in range(0, len(numAry), 1):
    if int(numAry[num] % 519) == 0:
        hap += numAry[num]

print("퀴즈3 : 777 부터 23455까지 519의 배수의 합 : ", hap)