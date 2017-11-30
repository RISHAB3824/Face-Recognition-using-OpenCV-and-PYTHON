import cv2
import numpy as np
import sys


facedetect=cv2.CascadeClassifier('E:\OpenCv and Python\opencv\sources\data\haarcascades/haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
 
while(True):
    ret,img=cap.read();
    if ret == True:
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces=facedetect.detectMultiScale(gray,scaleFactor=1.4,minNeighbors=5,minSize=(30,30),flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

      for(x,y,w,h) in faces:
          cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
      cv2.imshow('DetectFace',img)
      if (cv2.waitKey(2) == ord('q')):
          break
cam.release()
cv2.destroyAllWindows()

    
