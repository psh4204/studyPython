# 바이너리 파일 읽기 :  *.raw _ C:\images\meRAW\LENA256.raw
## 함수

## 변수
filename = "C:\\images\\meRAW\\me512.raw"

## 메인
rfp = open(filename, "rb")

pixel = int(ord(rfp.read(1))) # 1바이트 만 읽기. (바이너리코드는 1바이트씩 읽기 권장)
                        # ord() : 해당문자 유니코드화
                        # chr() : 해당숫자를 문자화
print(pixel)

rfp.close()