import sys
import os
import glob
import re
import numpy as np 

import tensorflow
import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.framework import ops
ops.reset_default_graph()

from flask import Flask, render_template, redirect, url_for, request
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app

app = Flask(__name__)

MODEL_PATH = "/Users/dipit/Desktop/AnimalClassifier/InceptionV3_model.h5"

model = load_model(MODEL_PATH)


def model_predict(img_path, model):
	IMG = image.load_img(img_path, target_size=(299,299))

	# Preprocessing on the image
	IMG = image.img_to_array(IMG)
	#IMG = np.array(IMG)
	IMG = np.true_divide(IMG, 255)
	#IMG = np.expand_dims(IMG, axis=0)
	IMG = IMG.reshape(1,299,299,3)
	print(IMG.shape)
	#IMG = preprocess_input(IMG, mode="caffe")

	model.compile(loss="categorical_crossentropy", metrics=['accuracy'], 
					optimizer="Adam")
	#predictions = model.predict(img)
	predictions_c = np.argmax(model.predict(IMG), axis=1)
	return predictions_c


# : : : Flask Routes
@app.route('/', methods=['GET'])

def index():
	return render_template('index.html')

@app.route("/predict",methods=['GET', 'POST'])

def upload():

	classes = {
    	'TRAIN': ['CHEETAH', 'HYENA', 'JAGUAR', 'TIGER'],
    	'VALIDATION': ['CHEETAH', 'HYENA', 'JAGUAR', 'TIGER']
	}

	if request.method == "POST":

		f = request.files['file']

		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads',secure_filename(f.filename))
		f.save(file_path)

		prediction = model_predict(file_path, model)

		predicted_class = classes['VALIDATION'][prediction[0]]
		print("We think that is a {}.".format(predicted_class.lower()))

		return str(predicted_class).lower()

if __name__ == "__main__":
	app.run(debug = True)












































