import cv2
import numpy as np


face_classifier = cv2.CascadeClassifier('E:\OpenCv and Python\opencv\sources\data\haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
count = 0


while True:

    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray,scaleFactor=1.4,minNeighbors=5,minSize=(30,30),flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
      #  if faces is ():
           #return None

        for(x,y,w,h) in faces:
            count += 1
            cropped_face = frame[y:y+h, x:x+w]
            cv2.rectangle (frame,(x,y),(x+w,y+h),(255,0,0),4)
            cv2.waitKey(100)
            if cropped_face is not None:
           
                face = cv2.resize(cropped_face, (200, 200))
                face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)

        
            file_name_path = 'images/User.' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, frame)

            cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Cropper', frame)
        
          #  else: 
            #  print("Face not found")
                #pass

            if cv2.waitKey(1) == 13 or count == 150: #13 is the Enter Key
                 
                 break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")
