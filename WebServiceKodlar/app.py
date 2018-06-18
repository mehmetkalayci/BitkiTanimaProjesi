import io
import os
import numpy as np
from flask import Flask, render_template,request, jsonify, send_from_directory
import flask
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing.image import img_to_array

model = load_model('model.h5')

#initialize flask app
app = Flask(__name__)

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # return the processed image
    return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if request.method == "POST" and request.files['image']:
		image = flask.request.files["image"].read()
		image = Image.open(io.BytesIO(image))
		
		image = prepare_image(image, target=(150, 150))
		# indicate that the request was a success
		data["success"] = True
		
		result = model.predict(image)
		
		data["class"] = str(result[0][0])
		
		if result[0][0] == 1:
			data["class_str"] = "Lale"
		elif result[0][0] == 0:
			data["class_str"] = "Papatya"
		else:
			data["class_str"] = "Tespit edilemedi!"
		
	return jsonify(data)

@app.route('/')
def index():
	return render_template("index.html")


if __name__ == "__main__":
	print("START FLASK")
	#load_model()

	port = int(os.environ.get('PORT', 5000))
	app.run(host='127.0.0.1', port=port)
