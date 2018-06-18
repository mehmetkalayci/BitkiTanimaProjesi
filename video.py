import numpy as np
import cv2
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing import image

cap = cv2.VideoCapture('C:/Users/Hatice/Desktop/BitkiTanima/tarla.mp4')
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# load saved model
model = load_model('model.h5')

while cap.isOpened():
	ret, frame = cap.read()
    # frame = frame[0:360, 0:480] # crop frame

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray
    # cv2.imshow('frame', gray)
	cv2.imwrite(filename="alpha.png", img=frame);  # write frame image to file
	# sizing the input to set as input to the convolution network
	img_pred = image.load_img("alpha.png", target_size=(150, 150))
	# convert img to array
	img_pred = image.img_to_array(img_pred)
	img_pred = np.expand_dims(img_pred, axis = 0)
	# prediction part
	result = model.predict(img_pred)
	# write the result to frame's  top left corner
	cv2.putText(img=frame, text='Tahmin: {}'.format("Lale" if result[0][0]==1 else "Papatya"), org=(10, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0))
	cv2.imshow('frame', frame)
	# write the result to console
	print('Tahmin: {} {}'.format("Lale" if result[0][0]==1 else "Papatya", result[0][0]))
	# quit if press down q key
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# release resources
cap.release()
cv2.destroyAllWindows()