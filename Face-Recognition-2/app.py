# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 21:26:39 2021

@author: Purav
"""

from flask import Flask
from flask import render_template
from flask import Response
import cv2
from PIL import Image
import numpy as np
from keras.models import load_model

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('facerecognition_model_vgg16.h5')

app = Flask(__name__)
camera = cv2.VideoCapture(0)


@app.route("/")
def index():
    return render_template("VideoStream.html")

def face_extractor(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        
        face=face_extractor(frame)
        if type(face) is np.ndarray:
            face = cv2.resize(face, (224, 224))
            im = Image.fromarray(face, 'RGB')
            #Resizing into 128x128 because we trained the model with this image size.
            img_array = np.array(im)
                    #Our keras model used a 4D tensor, (images x height x width x channel)
                    #So changing dimension 128x128x3 into 1x128x128x3 
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
        
                     
            name="None matching"
        
            if(pred[0]>0.5):
                name='Purav'
            cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video")
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

