# USAGE

# import of packets required for real-time video streaming
# import for identifying related packages

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# Instructions required to construct the command line
ap = argparse.ArgumentParser()

#Import Caffe package into identification process
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")

#for Caffe training
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")

#Filter conditions for low confidence recognition
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Initialize class labels and color different settings

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our model
# Print out load model
print("loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Initialize the video and turn on the camera sensor in advance
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#set the fps to zero
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# Get the key frame from the real-time video stream and set it to the specified output pixel size

	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# build the blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob and get the detections
	net.setInput(blob)
	detections = net.forward()

	f=open(r'C:\Users\47566\Desktop\recognition\re\1.txt','w')
	# loop for the detections
	for i in np.arange(0, detections.shape[2]):
		# get the confidence and use it for  comparison of the following confidence levels
		confidence = detections[0, 0, i, 2]

		# filter the lower confidence
		if confidence > args["confidence"]:
			# Get the class label we set at the beginning, and calculate the relevant coordinates
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Using cv to draw prediction results
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

			if  (confidence* 100) > 80:
		
			print("{}".format(label),file = f)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(key) 

	# set the q when we press ,the program will be closed
	if key == ord("q"):
		break

	# update the fps for printing
	fps.update()

# stop the fps
fps.stop()
#print the information of the video
print("elapsed time: {:.2f}".format(fps.elapsed()))
print("approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()