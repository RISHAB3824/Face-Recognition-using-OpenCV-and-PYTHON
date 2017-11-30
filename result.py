import cv2
import numpy as np
import sys


facedetect=cv2.CascadeClassifier('E:\OpenCv and Python\opencv\sources\data\haarcascades/haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
rec=cv2.createLBPHFaceRecognizer()
rec.load("rec\\train.yml")   
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,3,1,0,2)
while(True):
    ret,img=cap.read();
    if ret == True:
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces=facedetect.detectMultiScale(gray,scaleFactor=1.4,minNeighbors=5,minSize=(30,30),flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

      for(x,y,w,h) in faces:
          cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
          id,conf=rec.predict(gray[y:y+h,x:x+w])
          cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,(0,255,0))
      cv2.imshow('DetectFace',img)
      if (cv2.waitKey(2) == ord('q')):
          break
cam.release()
cv2.destroyAllWindows()

    
