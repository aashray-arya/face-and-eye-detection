#!/usr/bin/env python

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/home/user/haar_cascades/frontalface.xml')
eyes_cascade = cv2.CascadeClassifier('/home/user/haar_cascades/eye.xml')
        
img_original = cv2.imread('/home/user/img.jpg')#Enter the location of the photo
  

img_original = cv2.resize(img_original,(500,300))

img = cv2.resize(img_original,(500,300))       

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


ScaleFactor = 1.2

minNeighbors = 3

face = face_cascade.detectMultiScale(gray, ScaleFactor, minNeighbors)

for (x,y,w,h) in face:

	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  
	roi = img[y:y+h, x:x+w]
eyes = eyes_cascade.detectMultiScale(roi)
for (ex,ey,ew,eh) in eyes:
 	cv2.rectangle(roi, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)


cv2.imshow('Face_original',img_original)

cv2.imshow('Face',img)

cv2.waitKey(1)


cv2.waitKey(0)
cv2.destroyAllWindows()
