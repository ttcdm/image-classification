###MOSTLY CHATGPT GENERATED###
###IF IT DOESN'T WORK ON ALL RACES, SAY SOMETHING ABOUT IT
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

pygame.init()
screeen = pygame.display.set_mode((640, 640))

f = "model5.keras"
class_names = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
def pred(img):
  imgarr = keras.utils.img_to_array(img)
  imgarr = np.expand_dims(imgarr, axis=0)
  images = np.vstack([imgarr])
  model = keras.models.load_model(f)
  result = model.predict(images)[0]
  result = dict(zip(class_names, result))
  for i in result:
    #print(i, result[i])
    pass
  k = [result[i] for i in result]
  k, v = list(result.keys()), list(result.values())

  print(k[v.index(max(v))], max(v))


cap = cv2.VideoCapture(0)
#"""
img = None

while True:
    ret, frame=cap.read()
    cv2.imshow('frame', frame)
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            print(event)
            if event == pygame.K_q:
                pygame.quit()

                sys.exit()
                exit()
            if event.key == pygame.K_a:
                newframe=cv2.resize(frame, (48, 48))
                newframe=cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
                #img = keras.utils.load_img(newframe)
                pred(newframe)
                print("hi")
                #pred(newframe)
#"""
"""
ret, frame = cap.read()
if cv2.waitKey(1) & 0xFF == ord('q'):
    pass
newframe = cv2.resize(frame, (64, 64))
newframe = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
cv2.imshow('frame', frame)
"""
#img = keras.utils.load_img(newframe)
#img = keras.utils.load_img("test1.jpg", color_mode="grayscale", target_size = (48, 48))  # grayscale because it was trained on grayscale

#pred(img)
cap.release()
#cv2.destroyAllWin
exit()