import os
from PIL import Image
import numpy as np
import cv2 as cv
import pickle

import logging
logging.basicConfig(level=logging.DEBUG)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Faces")

face_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')

recognizer = cv.face.LBPHFaceRecognizer_create()



y_labels = []
x_train = []
current_id = 0
label_ids = {}

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("jfif") or file.endswith("PNG") or file.endswith("JPG") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1
            id_ = label_ids[label]
            print(label_ids)
            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, "uint8")
            print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=6)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


print(y_labels)
print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
