#!/usr/bin/env python

import numpy as np
import cv2
import time

face_cascade = cv2.CascadeClassifier('/home/user/haar_cascades/frontalface.xml')
eyes_cascade = cv2.CascadeClassifier('/home/user/haar_cascades/eye.xml')
roi=None

def detect_face_and_eyes(image_frame):

	img_original=image_frame
	
	img=image_frame

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	ScaleFactor = 1.2

	minNeighbors = 3

	face = face_cascade.detectMultiScale(gray, ScaleFactor, minNeighbors)

	for (x,y,w,h) in face:

		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  
		roi = img[y:y+h, x:x+w]

	if(roi is not None):
		eyes = eyes_cascade.detectMultiScale(roi)

		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	cv2.imshow('Face_original',img_original)
	cv2.imshow('Face',img)

def main():
    #video_capture = cv2.VideoCapture(0)  #for live tracking
    video_capture = cv2.VideoCapture('video_path') #add video path for tracking in a recorded video

    while(True):
        ret, frame = video_capture.read()
        detect_face_and_eyes(frame)
        time.sleep(0.033)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

cv2.waitKey(0)
cv2.destroyAllWindows()
