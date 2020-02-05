import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import os
from math import floor
import cv2
import matplotlib.pyplot as plt

images = []
labels = []
class_names = []
real_labels = []
unique = {}

# haarcascade for face detection in image and add path where your file download
# if you haven't then download it from here: https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier(r'face_recognition\save_model\haarcascade_frontalface_default.xml')

def load_images_from_folder(folder):
    # every image in the directory converted into numpy array
    a=1
    for file in os.listdir(folder):
        for filename in os.listdir(folder+"\\"+file):
            img = cv2.imread(folder+"\\"+file+"\\"+filename)
            if img.shape != (100,100,3):
                img = cv2.resize(img, (680, 680))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    real_image = img[y:y + h, x:x + w]
                    real_image = cv2.resize(real_image, (100, 100))
                img = real_image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = img // 255.0
            images.append(img)

            if file not in class_names:
                class_names.append(file)
            real_labels.append(file)

def unique_list():
    v = 0
    for i in class_names:
        if i not in unique:
            unique[i] = v
            v += 1

def labels__():
    for i in real_labels:
        labels.append(unique[i])

def dataset():
    load_images_from_folder(r"face_recognition\dataset")
    unique_list()
    labels__()
    leng = floor(len(images) * 85 / 100)
    # images and labels are split into 85% as train_images,train_labels,test_images,test_labels
    train_images = np.asarray(images[0:leng])
    train_labels = np.asarray(labels[0:leng])
    test_images = np.asarray(images[leng:])
    test_labels = np.asarray(labels[leng:])
    return train_images,train_labels,test_images,test_labels,class_names


def old_model(train_images,train_labels,test_images,test_labels):
    #neural network model is created and test it
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(100, 100)),
        keras.layers.Dense(500, activation="relu"),
        keras.layers.Dense(250, activation="relu"),
        keras.layers.Dense(125, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(len(class_names), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    model.fit(train_images, train_labels, epochs=5)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("test accuracy:", test_acc , "test_loss", test_loss)
    new_model = load_model(r"face_recognition/save_model/save_model.h5")
    del new_model
    model.save(r"face_recognition\save_model\save_model.h5")
    return model

def model_summary(model):
    # this function give summary of model
    model.summary()

def predict(test,class_names):
    # this function is use for predict persons name
    model = load_model(r"face_recognition/save_model/save_model.h5")
    prediction = model.predict(test)
    return class_names[np.argmax(prediction)]

