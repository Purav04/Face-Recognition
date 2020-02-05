import os
import cv2
from project_face_recognition import train_model as second
import numpy as np

# haarcascade for face detection in live video
# if you dont have it then download it from here: https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier(r'face_recognition\save_model\haarcascade_frontalface_default.xml')

# take train_images,train_labels,test_images,test_labels,class_names from
# image data from train_model
train_images,train_labels,test_images,test_labels,class_names = second.dataset()

# second.old_model(...) is for train our midel and save it
# it take train_images,train_labels,test_images,test_labels as argument
# model = second.old_model(train_images,train_labels,test_images,test_labels)

font = cv2.FONT_HERSHEY_PLAIN
cap = cv2.VideoCapture(0)


def load_images_from_folder(folder):
    # load total number of images in directory
    a=0
    for file in os.listdir(folder):
        for filename in os.listdir(folder+"\\"+file):
            a+=1
    return a


while cap.isOpened():
    ret,img = cap.read()
    # convert video into gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detect face in gray scale video
    faces = face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        # on detected face it draw a rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 3)
        real_image = img[y:y + h, x:x + w]
        real_image = cv2.resize(real_image, (100, 100))
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
        real_image = real_image // 255.0

        l = []
        l.append(real_image)
        l = np.asarray(l)
        # second.predict(...) find the person name from the save neural network model.
        # it take detected face and class_names as argument
        text = second.predict(l, class_names)
        del l

        # put text above the rectangle on the detected face
        img = cv2.putText(img, text, (x, y), font, 2,(0,0,255),1)

    cv2.imshow('video',img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key==ord('s'):
        # it count the total images in directory
        a = load_images_from_folder(r"face_recognition\opencv_images")
        # save detected face to a particular path
        cv2.imwrite((r'face_recognition/dataset/UNKNOWN/image{}.jpg').format(a+1), real_image)

cap.release()
cv2.destroyAllWindows()

