import numpy as np
import pandas as pd
import math
import os
import keras
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from keras.applications import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from tensorflow.python.framework import ops
ops.reset_default_graph()
import PIL
from PIL import Image
from glob import glob

train_data = "/users/dipit/AnimalClassifier/train/"
valid_data = "/users/dipit/AnimalClassifier/validation/"

img_cols, img_rows = 299, 299
num_classes = 3
num_channels = 3
batch_size = 50

incepnet = InceptionV3(input_shape=(img_rows, img_cols, num_channels),
                        weights="imagenet",
                        include_top=False)

for layer in incepnet.layers:
    layer.trainable = False

folders = glob('/users/dipit/AnimalClassifier/train/*')

x = Flatten()(incepnet.output)
prediction = Dense(len(folders), activation="softmax")(x)
model = Model(inputs=incepnet.input, outputs=prediction)
print(model.summary())

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode="nearest")

val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_data,
                                                    target_size=(img_rows, img_cols),
                                                    shuffle=True,
                                                    batch_size=batch_size,
                                                    class_mode="categorical")

val_generator = val_datagen.flow_from_directory(valid_data,
                                                target_size=(img_rows, img_cols),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                class_mode="categorical")

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs=5,
                    steps_per_epoch=len(train_generator),
                    validation_data=val_generator,
                    validation_steps=len(val_generator))


scores = model.evaluate_generator(val_generator, steps=len(val_generator),
                                  verbose=1)
print("Val Accuracy:%.3f Val Loss:%.3f"%(scores[1]*100, scores[0]))
model.save("InceptionV3_model.h5")

model = load_model("InceptionV3_model.h5")

plt.figure(figsize=(12,8))
plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=16)
plt.grid()
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title("Accuracy Curves")

plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=16)
plt.grid()
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Loss Curves")
plt.show()

val_pred = model.predict(val_generator, steps=len(val_generator),
                         verbose=1)
val_labels = np.argmax(val_pred)

