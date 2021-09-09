import cv2

src = cv2.imread("c:/images/aa.jpg")
print(src.ndim)
dst = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
print(dst.ndim)

cv2.imshow("Title1", src)
cv2.imshow("Title2", dst)

# [바로 닫히지 않게 멈추게 한다.]
cv2.waitKey(0)
cv2.destoryWindow()

