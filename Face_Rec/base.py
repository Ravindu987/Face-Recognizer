import numpy as np
import cv2
import pickle

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels={}
with open("labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_color=frame[y:y+h,x:x+w]
        roi_gray=gray[y:y+h,x:x+w]

        id_,conf=recognizer.predict(roi_gray)

        if conf>=30:
            #print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            cv2.putText(frame,name,(x,y),font,1,color,2,cv2.LINE_AA)
        else:
            font=cv2.FONT_HERSHEY_SIMPLEX
            color=(255,255,255)
            cv2.putText(frame,"Not identified",(x,y),font,1,color,2,cv2.LINE_AA)           

        color=(0,255,0)
        stroke=2
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()