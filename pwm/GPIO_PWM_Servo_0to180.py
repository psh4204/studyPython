import RPi.GPIO as GPIO
import time

# Note : https://velog.io/@psh4204/RaspberryPi-%EC%84%9C%EB%B3%B4%EB%AA%A8%ED%84%B0-%EC%A0%9C%EC%96%B4

# GPIO Servo모터 제어

servo_pin = 25
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)
pwm = GPIO.PWM(servo_pin, 50) # 50Hz( 서보모터 PWM 동작을 위한 주파수 )
pwm.start(3.0) # 서보모터의 0도 위치( 0.6ms ) 이동: 값 3.0은 pwm 주기인 20ms 의 3% 를 의미

scale = int(input("스케일 값을 입력하시오(1~10) : ")) # min = 1, max = 10
if 1 <= scale <= 10 :
    pass
else :
    scale = 1

try : 
    while True:
        for i in range(int(30*scale), int(125*scale),2) :
            # 0.02초간 0.019도 움직일 수 있음
            pwm.ChangeDutyCycle(i/int(10*scale))
            time.sleep(0.02)
        for i in range( int(125*scale), int(30*scale),-2) :
            pwm.ChangeDutyCycle(i/int(10*scale))
            time.sleep(0.02)

# Ctrl + C : 종료 및 GPIO 초기화
except KeyboardInterrupt:
    GPIO.cleanup() 
