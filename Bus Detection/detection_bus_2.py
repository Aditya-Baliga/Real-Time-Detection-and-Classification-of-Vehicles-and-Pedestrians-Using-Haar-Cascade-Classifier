# -*- coding: utf-8 -*-

import cv2
print('Project Topic : Vehicle Classification')
print('Research Internship on Machine learning using Images')
print('By Aditya Baliga B and Aditya Yogish Pai')

cascade_src = 'Bus_front.xml'
video_src = 'bus2.mp4'


cap = cv2.VideoCapture(video_src)
fgbg=cv2.createBackgroundSubtractorMOG2()
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, img = cap.read()
    fgmask=fgbg.apply(img)
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.16, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
