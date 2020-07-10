import numpy as np
import keras
import tensorflow.keras as keras
from keras.applications import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam
from tensorflow.python.framework import ops
ops.reset_default_graph()
import PIL
from PIL import Image
from glob import glob

model = load_model("InceptionV3_model.h5")
model.compile(loss="categorical_crossentropy",
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])

IMG = Image.open('/Users/dipit/Desktop/AnimalClassifier/test/test1.jpg')
IMG = IMG.resize((299, 299))
IMG = np.array(IMG)
IMG = np.true_divide(IMG, 255)
IMG = IMG.reshape(1, 299, 299, 3)
print(IMG.shape)

predictions = model.predict(IMG)
predictions_c = np.argmax(model.predict(IMG), axis=1)

classes = {
    'TRAIN': ['CHEETAH', 'HYENA', 'JAGUAR', 'TIGER'],
    'VALIDATION': ['CHEETAH', 'HYENA', 'JAGUAR', 'TIGER']
}
predicted_class = classes['VALIDATION'][predictions_c[0]]
print('We think that is {}.'.format(predicted_class.lower()))