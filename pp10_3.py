import matplotlib.pyplot as plt
import numpy as np
#from silence_tensorflow import silence_tensorflow
#silence_tensorflow("ERROR")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.keras.config.disable_interactive_logging()
#import tarfile as tar
#from pp10_2 import train
#maybe use zipfile
#try to find fix for spam warnings i think when the gpu runs

batch_size = 64
img_height = 500
img_width = 500
epochs = 40
f = "model11.keras"

extr = 0
mode = "test"#train / test
device = "GPU"


print(tf.config.list_physical_devices('GPU'))

"""
from tensorflow.compat.v1 import ConfigProto#memory growth
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

gpus = tf.config.list_physical_devices('GPU')
if gpus:#allocate a certain amount of memory
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=7500)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
"""

"""
just add more tpaths to use different folders to train the model.
you can also do the same for the vpath but i'm not sure if the validation actually changes the model"""

tpath = "D:\\new ai dataset\\New happy sad data set black and white\\images\\train"
tpath1 = "D:\\new ai dataset\\New happy sad data set black and white\\new data set"
tpath2 = "G:\\.shortcut-targets-by-id\\1DncfwZdakbRXDZOR5ZUdO2do0_hLaYhr\\archive"
tpath3 = "C:\\Users\\Titus\\Downloads\\higher dataset"
vpath = "D:\\new ai dataset\\New happy sad data set black and white\\images\\validation"
tpath4 = "/mnt/c/Users/Titus/Downloads/ds/train"
#tpath4 = "C:\\Users\\Titus\\Downloads\\ds\\train"
tpath5 = "C:\\Users\\Titus\\Downloads\\fruits.zip"
tpath5 = "/mnt/c/Users/Titus/pycharmprojects/image-classification/fruits/fruits-360_dataset_100x100/fruits-360/Training"
tpath6 = "C:\\Users\\Titus\\Downloads\\natural images.zip"
tpath6 = "/mnt/c/Users/Titus/pycharmprojects/image-classification/natural_scenes/seg_train/seg_train"

if extr == 1:
    from zipfile import ZipFile
    with ZipFile(tpath6, 'r') as z:
        z.extractall(path="natural_scenes")

train_ds = tf.keras.utils.image_dataset_from_directory(
    tpath6,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode = "rgb"#grayscale
)
print(train_ds.class_names)

val_ds = tf.keras.utils.image_dataset_from_directory(
   tpath6,
   validation_split=0.2,
   subset="validation",
   seed=123,
   image_size=(img_height, img_width),
   batch_size=batch_size,
   color_mode = "rgb"
)

def train(train_ds, val_ds, class_names):
    #AUTOTUNE = tf.data.AUTOTUNE
    #train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)#somehow these lines cause the memory to max out
    #val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)#this and upwards was commented out for earlier stuff to make it more original or something
    normalization_layer = layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    #print(np.min(first_image), np.max(first_image))

    num_classes = len(class_names)

    model = keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(padding = "same"),#added padding = "same" to the maxpooling2d to prevent negative dimension size error
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(padding = "same"),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(padding = "same"),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
  ])

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    try:
        model = keras.models.load_model(f)
    except Exception as e:
        print(e)
    #with tf.device("/GPU:0"):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    try:
        model.save(f)
    except Exception as ee:
        print(ee)

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

def testm(dir):
    model = 0
    try:
        model = keras.models.load_model(f)
    except Exception as e:
        print(e)
        pass
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dir,
        #validation_split=0.2,
        #subset="",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode = "rgb"
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

if mode == "train":
    with tf.device(f"/{device}:0"):
        class_names = train_ds.class_names
        """
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
        #plt.show()#HERE
        """
        train(train_ds, val_ds, class_names)
elif mode == "test":
    testm("/mnt/c/Users/Titus/pycharmprojects/image-classification/natural_scenes/seg_test/seg_test")

#img = keras.utils.load_img("D:\\new ai dataset\\New happy sad data set black and white\\other\\happy\\test.jpg", color_mode="grayscale", target_size = (img_height, img_width))  # grayscale because it was trained on grayscale
#pred(img)

