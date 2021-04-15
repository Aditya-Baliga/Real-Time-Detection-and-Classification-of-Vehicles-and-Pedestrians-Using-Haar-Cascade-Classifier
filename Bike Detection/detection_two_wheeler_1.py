# -*- coding: utf-8 -*-

import cv2
print('Project Topic : Vehicle Classification')
print('Research Internship on Machine learning using Images')
print('By Aditya Baliga B and Aditya Yoggish Pai')

cascade_src = 'two_wheeler.xml'

video_src = 'two_wheeler2.mp4'

cap = cv2.VideoCapture(video_src)
fgbg = cv2.createBackgroundSubtractorMOG2()
car_cascade = cv2.CascadeClassifier(cascade_src)


while True:
    ret, img = cap.read()
    fgbg.apply(img)
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray,1.01, 1)


    for (x_axis, y_axis, width, height) in cars:
        cv2.rectangle(img, (x_axis, y_axis), (x_axis+width, y_axis+height), (0,255,215), 2)
    
    cv2.imshow('Two wheeler detection 1', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
