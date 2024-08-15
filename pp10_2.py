import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
batch_size = 128
img_height = 128
img_width = 128
epochs = 100
f = "model6.keras"

"""
just add more tpaths to use different folders to train the model.
you can also do the same for the vpath but i'm not sure if the validation actually changes the model"""

tpath = "D:\\new ai dataset\\New happy sad data set black and white\\images\\train"
tpath1 = "D:\\new ai dataset\\New happy sad data set black and white\\new data set"
tpath2 = "G:\\.shortcut-targets-by-id\\1DncfwZdakbRXDZOR5ZUdO2do0_hLaYhr\\archive"
tpath3 = "C:\\Users\\Titus\\Downloads\\higher dataset"
vpath = "D:\\new ai dataset\\New happy sad data set black and white\\images\\validation"

"""
train_ds = tf.keras.utils.image_dataset_from_directory(
  tpath,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode = "grayscale"
)
print(train_ds.class_names)

train_ds1 = tf.keras.utils.image_dataset_from_directory(
  tpath1,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode = "grayscale"
)
print(train_ds1.class_names)
"""
train_ds2 = tf.keras.utils.image_dataset_from_directory(
  tpath3,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode = "grayscale"
)
print(train_ds2.class_names)
"""

val_ds = tf.keras.utils.image_dataset_from_directory(
  tpath,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode = "grayscale"
)
val_ds1 = tf.keras.utils.image_dataset_from_directory(
  tpath1,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode = "grayscale"
)
"""
val_ds2 = tf.keras.utils.image_dataset_from_directory(
  tpath3,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode = "grayscale"
)



def train(train_ds, val_ds, class_names):
  #AUTOTUNE = tf.data.AUTOTUNE
  #train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  #val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
  normalization_layer = layers.Rescaling(1./255)
  normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
  image_batch, labels_batch = next(iter(normalized_ds))
  first_image = image_batch[0]
  print(np.min(first_image), np.max(first_image))

  num_classes = len(class_names)

  model = keras.Sequential([
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),#padding = "same"),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),#padding = "same"),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),#padding = "same"),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
  ])
  model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  try:
    model = keras.models.load_model(f)
  except Exception as e:
    print(e)
    pass

  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )

  from tensorflow.keras import models

  try:
    model.save(f)
  except Exception as ee:
    print(ee)
    pass

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)
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
  plt.show()#HERE

def testm():
  model = 0
  try:
    model = keras.models.load_model(f)
  except Exception as e:
    print(e)
    pass
  test_ds = tf.keras.utils.image_dataset_from_directory(
    "D:\\new ai dataset\\New happy sad data set black and white\\new data set",
    #validation_split=0.2,
    #subset="",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode = "grayscale"
  )
  res = model.evaluate(test_ds)
  print(res)

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


#class_names = train_ds.class_names
#train(train_ds, val_ds, class_names)
#"""
#class_names = train_ds1.class_names
#train(train_ds1, val_ds1, class_names)
class_names = train_ds2.class_names
train(train_ds2, val_ds2, class_names)
#testm()
#"""
#img = keras.utils.load_img("D:\\new ai dataset\\New happy sad data set black and white\\other\\happy\\test.jpg", color_mode="grayscale", target_size = (img_height, img_width))  # grayscale because it was trained on grayscale
#pred(img)