# -*- coding: utf-8 -*-

import cv2
print('Project Topic : Vehicle Classification')
print('Research Internship on Machine learning using Images')
print('Press escape key to exit the application')

# is the trained two wheeler haar cascade file
twoWheelerHaarCascadePath = 'two_wheeler.xml'
# is the test video on which the detection algorithm is to be run
testVideoSource = 'two_wheeler2.mp4'

# here we set the VideoCapture source as testVideoSource
cap = cv2.VideoCapture(testVideoSource)

# here we set & create the MOG2 background subtractor instance
fgbg = cv2.createBackgroundSubtractorMOG2()

twoWheelerCascade = cv2.CascadeClassifier(twoWheelerHaarCascadePath)


while True:
    # here returnStatus holds true/ false, meaning able to capture/read the frame from video, 
    # img is the frame captured
    returnStatus, img = cap.read()
    if (returnStatus == False):
        continue;
    # if image img captured is null, exit the loop
    if (type(img) == type(None)):
        break
    
    # we could use foreground extractor background subtractor in videos
    # where background is still to improve region of interest for better detection
    # fmask = fgbg.apply(img)
    # cv2.imshow('with masking', fmask )

    # used to gray scale the image for better performance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # here individual item in the list corresponds to [x_axis, y_axis, width, height] of the detected object
    # if list has multiple items, it means multiple objects where detected 
    # order of list is m X n, where m- rows ie no of objects detected in the frame and n = 4 ie co-ordinates
    # detectMultiScale takes (image frame, scaling factor, no of neighbours )
    detectedTwoWheelerList = twoWheelerCascade.detectMultiScale(gray,1.01, 1)

    if (len(detectedTwoWheelerList) == 0):
        continue

    # Drawing rectangle(s) for above detected objects in the current frame
    # here cv2.rectangle takes ( soure image frame, top left indices, bottom right indices, bgr color, rectangle border width)
    for (x_axis, y_axis, width, height) in detectedTwoWheelerList:
        cv2.rectangle(img, (x_axis, y_axis), (x_axis+width, y_axis+height), (0,255,215), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'Aditya Baliga', (10,50), font, 1, (200,255,255), 2, cv2.LINE_AA)
    # imshow is used to display the output tile, first parameter is the title of the tile and second param is the image img
    cv2.imshow('Two wheeler detection 1', img)
    
    # here 33 is the wait time in ms and 27 is the keycode for escape key, used to exit the loop
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
