# 파일처리
## 함수

## 전역변수

## 메인함수

rfp = open("../Day01/first.py", "r", encoding="utf-8")
wfp = open("test.txt", "w")

#
# lines = rfp.readlines()
# for line in lines :
#     print(line,end="")

#
# while True:
#     line = rfp.readline()
#     if(line== "" or line == None):
#         break
#     wfp.writelines(line)

rfp.close()
wfp.close()