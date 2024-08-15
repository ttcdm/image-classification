
import matplotlib.pyplot as plt
import numpy as np
print(np.__version__)
from PIL import Image
import tensorflow as tf

from tensorflow import keras



from tensorflow.keras import layers
#from tensorflow.keras.models import Sequential
import pathlib

"""
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
happy = list(data_dir.glob('happy/*'))
print(happy)
#Image._show(Image.open(str(roses[0])))
"""

#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
#data_dir = pathlib.Path(data_dir).with_suffix('')

#localimg = "D:\\new ai dataset\\New happy sad data set black and white\\images"
#data_dir = pathlib.Path(localimg).with_suffix('')


p = "D:\\new ai dataset\\New happy sad data set black and white\\images.zip"



batch_size = 32
img_height = 48
img_width = 48

train_ds = tf.keras.utils.image_dataset_from_directory(
  "D:\\new ai dataset\\New happy sad data set black and white\\images\\train",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "D:\\new ai dataset\\New happy sad data set black and white\\images\\validation",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)

"""#this just displays the first 9 images or something
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
"""
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
#plt.show()


#exit()#HERE

def train(train_ds, val_ds, class_names):
  AUTOTUNE = tf.data.AUTOTUNE

  train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

  normalization_layer = layers.Rescaling(1./255)

  normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
  image_batch, labels_batch = next(iter(normalized_ds))
  first_image = image_batch[0]
  # Notice the pixel values are now in `[0,1]`.
  print(np.min(first_image), np.max(first_image))

  num_classes = len(class_names)

  model = keras.Sequential([
    #layers.RandomFlip('horizontal', input_shape=(img_height, img_width, 3)),#image randomization to prevent overfitting
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),

    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)

  ])
  #model.load_weights("model1.keras")


  model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  #model.summary()#HERE

  #from keras.models import load_model

  try:
    model = keras.models.load_model("model2.keras")
  except Exception as e:
    #print("no weights found", RuntimeError, TypeError, NameError)
    print("unable to load weights")
    print(e)
    pass
  epochs = 6

  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )

  from tensorflow.keras import models

  try:
    model.save("model2.keras")
  except Exception as ee:
    #print("unable to save model", RuntimeError, TypeError, NameError)
    print("unable to save weights")
    print(ee)
    pass

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)
  """
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  #plt.show()#HERE
  """

  #result = model.evaluate(train_ds, val_ds)
  #print(result)

def testm():
  model = 0
  try:
    model = keras.models.load_model("model2.keras")
  except Exception as e:
    print("unable to load weights")
    print(e)
    pass
  try:
    img = keras.utils.load_img("D:\\new ai dataset\\New happy sad data set black and white\\other\\test\\test.jpg", color_mode = "grayscale")#grayscale because it was trained on grayscale

    #cimg = keras.models.Sequential([layers.Resizing(height = 48, width = 48), keras.layers.CenterCrop(height = 48, width = 48)])
    #for i in range(len(cimg.layers)):
    #  x = cimg.layers[i].output()
    #  print(x)

    inarr = keras.utils.img_to_array(img)

    inarr = np.array([inarr])
    inarr = np.repeat(inarr, 3, axis = -1)
    #acimg = cimg.fit(inarr)
    #print(acimg)
    #acimg = acimg.reshape(-1, 48, 48, 3)

    result = model.predict(inarr)
    print(result)
  except:
    pass

  test_ds = tf.keras.utils.image_dataset_from_directory(
    "D:\\new ai dataset\\New happy sad data set black and white\\new data set",
    #validation_split=0.2,
    #subset="",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
  )

  res = model.evaluate(test_ds)
  print(res)
#train(train_ds, val_ds, class_names)
testm()

