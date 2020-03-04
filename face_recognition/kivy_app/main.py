from kivy.app import App
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import numpy as np
import cv2
import os
from face_rec_app import train_model as second

#for facedetection and recognition in app 
class kivycamera(Image):
    def __init__(self, cap, fps, **kwargs):
        super(kivycamera, self).__init__(**kwargs)
        self.capture = cap
        self.face_cascade = cv2.CascadeClassifier("path of cascade file")
        train_images, train_labels, test_images, test_labels, self.class_names = second.dataset()
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        try:
            ret, frame = self.capture.read()
            img = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                # on detected face it draw a rectangle
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                real_image = img[y:y + h, x:x + w]
                real_image = cv2.resize(real_image, (100, 100))
                real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
                real_image = real_image // 255.0
                self.img_cap_save = real_image
                l = []
                l.append(real_image)
                l = np.asarray(l)
                self.text = second.predict(l, self.class_names)
                del l
                font = cv2.FONT_HERSHEY_PLAIN
                frame = cv2.putText(img, self.text, (x, y), font, 2, (0, 0, 255), 1)
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')  # frame.shape[0], frame.shape[0]
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = image_texture

        except:
            pass

#for open a camera in app
class open_cam(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        
        layout1 = BoxLayout(orientation='horizontal', size=(850, 50), size_hint=(None, None))
        self.btn2 = Button(text="images",
                           font_size="20sp",
                           background_color=(1,1,1,1),
                           color=(1, 1, 1, 1),
                           size=(500, 30), pos=(250, 100))
        self.btn2.bind(on_press=self.to_file_class)
        layout1.add_widget(self.btn2)

        self.add_widget(layout1)
        self.capture = cv2.VideoCapture(0)
        self.mycamera = kivycamera(self.capture, fps=1.0)
        self.add_widget(self.mycamera)

    def to_file_class(self,*_):
        camapp.screen_manager.current = "second"

#dataset of face recognition
class file(BoxLayout):
    def __init__(self,**kwargs):
        super().__init__(orientation='vertical',**kwargs)

        layout_ = BoxLayout(orientation='horizontal', size=(50, 50), size_hint=(None, None))
        self.btn = Button(text="<--",font_size="20sp",
                           background_color=(1,1,1,1),
                           color=(1, 1, 1, 1),
                           size=(500, 30), pos=(250, 100))
        self.btn.bind(on_press=self.go_back)
        layout_.add_widget(self.btn)
        self.add_widget(layout_)
        a=0
        self.l_i = []
        self.l_j = []
        for i in os.listdir(r"root folder path of dataset"):
            self.l_i.append(i)
            for j in os.listdir(r"dataset folder\{}".format(i)):
                a+=1
            self.l_j.append(a)
            a=0

        self.btn1 = Button(text = str(self.l_i[0]+"("+str(self.l_j[0])+" images )"),
                          font_size="20sp",
                          background_color=(1,1,1,1),
                         halign = "left")
        self.add_widget(self.btn1)

        self.btn2 = Button(text=str(self.l_i[1]+"("+str(self.l_j[1])+" images )"),
                           font_size="20sp",
                           background_color=(1,1,1,1),
                           halign="left")
        # self.btn.bind(on_press = self.inside)
        self.add_widget(self.btn2)

        self.btn3 = Button(text=str(self.l_i[2]+"("+str(self.l_j[2])+" images )"),
                           font_size="20sp",
                           background_color=(1,1,1,1),
                           halign="left")
        self.add_widget(self.btn3)

        self.btn4 = Button(text=str(self.l_i[3]+"("+str(self.l_j[3])+" images )"),
                           font_size="20sp",
                           background_color=(1,1,1,1),
                           halign="left")
        self.add_widget(self.btn4)

        self.btn5= Button(text=str(self.l_i[4]+"("+str(self.l_j[4])+" images )"),
                           font_size="20sp",
                           background_color=(1,1,1,1),
                           halign="left")
        self.add_widget(self.btn5)

        self.btn6 = Button(text=str(self.l_i[5]+"("+str(self.l_j[5])+" images )"),
                           font_size="20sp",
                           background_color=(1,1,1,1),
                           halign="left")
        self.add_widget(self.btn6)

        self.btn7 = Button(text=str(self.l_i[6]+"("+str(self.l_j[6])+" images )"),
                           font_size="20sp",
                           background_color=(1,1,1,1),
                           halign="left")
        self.add_widget(self.btn7)

    def go_back(self,*_):
        camapp.screen_manager.current = "first"


#build a app
class myapp(App):
    def build(self):
        self.screen_manager = ScreenManager()

        self.open_camera = open_cam()
        screen1 = Screen(name="first")
        screen1.add_widget(self.open_camera)
        self.screen_manager.add_widget(screen1)

        self.file = file()
        screen2 = Screen(name="second")
        screen2.add_widget(self.file)
        self.screen_manager.add_widget(screen2)

        return self.screen_manager

    def show_img(self,*_):
        pass


if __name__ == '__main__':
    camapp = myapp()
    camapp.run()
    cv2.destroyAllWindows()
