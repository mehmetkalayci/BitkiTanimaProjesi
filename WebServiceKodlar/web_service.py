import io
import os
import numpy as np
from flask import Flask, render_template,request, jsonify, send_from_directory
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing import image


model = load_model('model.h5')

#initialize flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if request.method == "POST" and request.files['image']:
		imagefile = request.files["image"].read()
		image = Image.open(io.BytesIO(imagefile))
		# indicate that the request was a success
		data["success"] = True
		
		img_pred = image.img_to_array(image)
		img_pred = np.expand_dims(img_pred, axis = 0)
		result = model.predict(img_pred)
		
		data["result"] = result[0][0]
		data["result_str"] = 'Tahmin: {}'.format("Lale" if result[0][0]==1 else "Papatya"
	# return the data dictionary as a JSON response
	return jsonify(data)


@app.route('/')
def index():
	return render_template("index.html")


if __name__ == "__main__":
	print("START FLASK")
	#load_model()

	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)