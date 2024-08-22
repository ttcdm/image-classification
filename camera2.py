###IF IT DOESN'T WORK ON ALL RACES, SAY SOMETHING ABOUT IT
import cv2
import numpy as np
import pygame
import sys
from tensorflow import keras

pygame.init()
screeen = pygame.display.set_mode((48, 48))


rel = 48

f = "model8.keras"
class_names = ["happy", "neutral", "sad"]
def pred(img):
  imgarr = keras.utils.img_to_array(img)
  imgarr = np.expand_dims(imgarr, axis=0)
  images = np.vstack([imgarr])
  model = keras.models.load_model(f)
  result = model.predict(images)[0]
  result = dict(zip(class_names, result))
  for i in result:
    print(i, result[i])
    pass
  k = [result[i] for i in result]
  k, v = list(result.keys()), list(result.values())
  print(k[v.index(max(v))], max(v))


cap = cv2.VideoCapture(0)
#"""
img = None

while True:
    ret, frame=cap.read()
    newframe = cv2.resize(frame, (rel, rel))
    newframe = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            print(event)
            if event == pygame.K_q:
                pygame.quit()
                sys.exit()
                exit()
            if event.key == pygame.K_a:
                newframe=cv2.resize(frame, (rel, rel))
                newframe=cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
                pred(newframe)

#img = keras.utils.load_img(newframe)
#img = keras.utils.load_img("test1.jpg", color_mode="grayscale", target_size = (48, 48))  # grayscale because it was trained on grayscale

#pred(img)
cap.release()
#cv2.destroyAllWin
exit()