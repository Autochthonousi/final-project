# encoding:utf-8
# USAGE
#coding:utf-8
#--coding:GBK -- 
# import the packages for the recognition
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2

def decode_predictions(scores, geometry):
	# Initializing the bounding box and initializing the confidence
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop for rows
	for y in range(0, numRows):
		# Bring up the score 
		scoresData = scores[0, 0, y]

		# set the coordinates of the text
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop for columns
		for x in range(0, numCols):
			#If the relationship between test scores and confidence is lower than the minimum confidence, it will not be recognized
			if scoresData[x] < args["min_confidence"]:
				continue

			# As mentioned in our report, setting a quadruple offset for our features
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# get the  data of the  image about angle, cos value and sin value
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# get the box height and width, which we need to use after
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# get the coordinates of the begining and ending, and we can use the data for the  recognition

			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return the rects and confidence we set at the beginning
	return (rects, confidences)

# Instructions required to construct the command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320,
	help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

#get the image and the data of the image by cv
image = cv2.imread(args["image"])
orig = image.copy()
(origH, origW) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
#get a new size of the picture, and get the ratio of these
(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)

# use cv to get the new picture size which we set before
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# we can set the name of the EAST output, for this project, we set the probabilities and corrdinate
layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]

# load the EAST model
print("loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])


#bulid a blob by picture and get the output which we metioned before 
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# get the  prediction of the decode and remove the  overlap part

(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

# reset the result list
results = []

# loop for box
for (startX, startY, endX, endY) in boxes:
	# get the different basic the ratio which we set
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	# padding around of the box to get the better result
	dX = int((endX - startX) * args["padding"])
	dY = int((endY - startY) * args["padding"])

	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(origW, endX + (dX * 2))
	endY = min(origH, endY + (dY * 2))

	# get the roi which we padding
	roi = orig[startY:endY, startX:endX]

	config = ("-l eng --oem 1 --psm 7")
	text = pytesseract.image_to_string(roi, config=config)

	results.append(((startX, startY, endX, endY), text))

#  get the result at the last
results = sorted(results, key=lambda r:r[0][1])


# loop for results
f=open(r'C:\Users\47566\Desktop\recognition\1.txt','w')
for ((startX, startY, endX, endY), text) in results:
# show the  OCR which we use the tesseract
	print("{}\n".format(text))

	print("{}\n".format(text),file = f)

f.close()