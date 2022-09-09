import numpy as np
import cv2 as cv
import pickle
from datetime import datetime

face_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
profile_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_profileface.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv.VideoCapture(0)
#cap = cv.VideoCapture("Videos/wolf.mp4")
#cap = cv.VideoCapture("Videos/django.mp4")

while(True):

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    ret, frame = cap.read()
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    centerWidth = 0.5*width
    centerHeight = 0.5*height

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y + h, x:x + w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >=70:
            print(id_)
            print(labels[id_])
            font = cv.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv.putText(frame, name, (x,y), font, 1, color, stroke, cv.LINE_AA)
        if conf <=70:
            font = cv.FONT_HERSHEY_SIMPLEX
            name = "undetermined"
            color = (255, 255, 255)
            stroke = 2
            cv.putText(frame, name, (x,y), font, 1, color, stroke, cv.LINE_AA)
        img_item_gray = "my-face-gray" + current_time + ".png"
        img_item_color = "my-face-color" + current_time + ".png"
        cv.imwrite(img_item_gray, roi_gray)
        cv.imwrite(img_item_color, roi_color)

        color = (255, 0, 0)
        stroke = 2
        cv.rectangle(frame, (x,y),(x+w,y+h), color, stroke)
        cv.rectangle(frame, (x+ int(0.49*w), y+int(0.49*h)), (x+int(0.51*w), y+int(0.51*h)), color, stroke)
        if x + 0.5 * w < centerWidth:
            print("move right")
        if x + 0.5 * w > centerWidth:
            print("move left")
        if y + 0.5 * h < centerHeight:
            print("move down")
        if y + 0.5 * h > centerHeight:
            print("move up")

    for (x, y, w, h) in profiles:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y + h, x:x + w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 40 and conf <= 105:
            print(id_)
            print(labels[id_])
            font = cv.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv.putText(frame, name, (x,y), font, 1, color, stroke, cv.LINE_AA)
        img_item_gray = "my-profile-gray.png"
        img_item_color = "my-profile-color.png"
        cv.imwrite(img_item_gray, roi_gray)
        cv.imwrite(img_item_color, roi_color)

        color = (0, 255, 0)
        stroke = 2
        cv.rectangle(frame, (x,y),(x+w,y+h), color, stroke)

    cv.imshow('frame', frame)
    #cv.imshow('gray', gray)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()