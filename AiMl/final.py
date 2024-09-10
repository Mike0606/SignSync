import cv2, pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load gesture recognition model
model = load_model('cnn_model_keras2.h5')

# Function to get the hand histogram
def get_hand_hist():
	with open("histograms.pkl", "rb") as f:
		hist = pickle.load(f)
	return hist[-1]

# Function to get image size for resizing
def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

# Preprocess image for Keras model
def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

# Make prediction using the model
def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

# Get the gesture label from the database
def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

# Get the predicted text from hand contour
def get_pred_from_contour(contour, thresh):
	x1, y1, w1, h1 = cv2.boundingRect(contour)
	save_img = thresh[y1:y1+h1, x1:x1+w1]
	text = ""
	if w1 > h1:
		save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
	elif h1 > w1:
		save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
	pred_probab, pred_class = keras_predict(model, save_img)
	if pred_probab*100 > 70:
		text = get_pred_text_from_db(pred_class)
	return text

# Get the contour and thresholded image
def get_img_contour_thresh(img):
	img = cv2.flip(img, 1)
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower_skin = np.array([0, 20, 70], dtype=np.uint8)
	upper_skin = np.array([20, 255, 255], dtype=np.uint8)
	mask = cv2.inRange(imgHSV, lower_skin, upper_skin)
	mask = cv2.GaussianBlur(mask, (5, 5), 0)
	thresh = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
	thresh = cv2.merge((thresh, thresh, thresh))
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	thresh = thresh[100:400, 300:600]  # Adjusted for region of interest
	contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
	return img, contours, thresh

# Function for speech output
def say_text(text):
	while engine._inLoop:
		pass
	engine.say(text)
	engine.runAndWait()

# Text recognition mode
def text_mode(cam):
	global is_voice_on
	text = ""
	word = ""
	count_same_frame = 0
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (640, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_text = text
		if len(contours) > 0:
			contour = max(contours, key=cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				text = get_pred_from_contour(contour, thresh)
				if old_text == text:
					count_same_frame += 1
				else:
					count_same_frame = 0

				if count_same_frame > 10:  # Faster detection threshold
					Thread(target=say_text, args=(text,)).start()
					word += text
					count_same_frame = 0

		# Display the result
		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		cv2.putText(blackboard, "Text Mode", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
		cv2.putText(blackboard, "Predicted text- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
		cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		cv2.rectangle(img, (300, 100), (600, 400), (0, 255, 0), 2)
		res = np.hstack((img, blackboard))
		cv2.imshow("Recognizing gesture", res)
		cv2.imshow("thresh", thresh)

		# Break on 'q' keypress
		if cv2.waitKey(1) == ord('q'):
			break

# Main function for gesture recognition
def recognize():
	cam = cv2.VideoCapture(1)
	if cam.read()[0] == False:
		cam = cv2.VideoCapture(0)

	while True:
		text_mode(cam)

# Initial prediction to warm up model
keras_predict(model, np.zeros((50, 50), dtype=np.uint8))

# Start recognition
recognize()
