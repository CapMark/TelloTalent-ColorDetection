import cv2
from PIL import Image
import numpy as np
from threading import Thread
from djitellopy import Tello
import time
global dir
h_min = 150
h_max = 179
s_min = 100
s_max = 255
v_min = 100
v_max = 255
width = 640
height = 480
deadZone = 100
frameWidth = width
frameHeight = height
lower = np.array([h_min, s_min, v_min])
upper = np.array([h_max, s_max, v_max])
me = Tello("192.168.1.140")
me.connect()
print(me.get_battery())
me.streamon()
def display(img):
    cv2.line(img,(int(frameWidth/2)-deadZone,0),(int(frameWidth/2)-deadZone,frameHeight),(255,255,0),3)
    cv2.line(img,(int(frameWidth/2)+deadZone,0),(int(frameWidth/2)+deadZone,frameHeight),(255,255,0),3)
    cv2.circle(img,(int(frameWidth/2),int(frameHeight/2)),5,(0,0,255),5)
    cv2.line(img, (0,int(frameHeight / 2) - deadZone), (frameWidth,int(frameHeight / 2) - deadZone), (255, 255, 0), 3)
    cv2.line(img, (0, int(frameHeight / 2) + deadZone), (frameWidth, int(frameHeight / 2) + deadZone), (255, 255, 0), 3)

def get_command(img,x1, x2, y1, y2):
    global dir
    dir=0
    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)
    if (cx <int(frameWidth/2)-deadZone):
        cv2.putText(img, " GO LEFT " , (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
        cv2.rectangle(img,(0,int(frameHeight/2-deadZone)),(int(frameWidth/2)-deadZone,int(frameHeight/2)+deadZone),(0,0,255),cv2.FILLED)
        dir = 1

    elif (cx > int(frameWidth / 2) + deadZone):
        cv2.putText(img, " GO RIGHT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
        cv2.rectangle(img,(int(frameWidth/2+deadZone),int(frameHeight/2-deadZone)),(frameWidth,int(frameHeight/2)+deadZone),(0,0,255),cv2.FILLED)
        dir = 2

    elif (cy < int(frameHeight / 2) - deadZone):
        cv2.putText(img, " GO UP ", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
        cv2.rectangle(img,(int(frameWidth/2-deadZone),0),(int(frameWidth/2+deadZone),int(frameHeight/2)-deadZone),(0,0,255),cv2.FILLED)
        dir = 3

    elif (cy > int(frameHeight / 2) + deadZone):
        cv2.putText(img, " GO DOWN ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 3)
        cv2.rectangle(img,(int(frameWidth/2-deadZone),int(frameHeight/2)+deadZone),(int(frameWidth/2+deadZone),frameHeight),(0,0,255),cv2.FILLED)
        dir = 4

    else: dir=0

def receive_video():
    global dir
    dir = 0
    tello_address = ('192.168.168.84', 11111)
    cap = cv2.VideoCapture(f'udp://{tello_address[0]}:{tello_address[1]}')
    while True:
        ret, vid = cap.read()
        img = cv2.resize(vid, (width, height))
        frame = img.copy()
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsvImage, lower, upper)
        #mask = cv2.medianBlur(mask, 15)
        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            get_command(frame, x1, x2, y1, y2)
            if dir!=0:
                print(dir)
        display(frame)
        if not ret:
            break
        cv2.imshow('Tello Talent Video Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            me.land()
            break
    cap.release()
    cv2.destroyAllWindows()
video_thread = Thread(target=receive_video)
video_thread.start()
me.takeoff()





