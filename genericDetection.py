from __future__ import print_function
from imutils.object_detection import non_max_suppression
#from google.colab.patches import cv2_imshow
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import argparse
# construct the argument parse and parse the argument
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, type=str, help="input the video path")
ap.add_argument("-o", "--output", required=True, type=str, help="Input the output video path")
args = vars(ap.parse_args())


writer = None
video = cv2.VideoCapture(args["input"])
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
H = None
W = None

def non_max_suppression_fast(boxes, overlapThresh):
    	# if there are no boxes, return an empty list
	if boxes.size == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
    print(boxes)
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")





while video.isOpened():
    ret, image = video.read()        
        
    if ret:
        image = imutils.resize(image, width=min(400, image.shape[1]))
        orig = image.copy()
        
        if W is None and H is None:
            (H, W) = image.shape[:2]        
        
        if writer is None:
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True )
        
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
        
        for (x, y, w, h) in rects:
        		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
      
       
            
        rects = np.array([x, y, x + w, y + h] for (x, y, w, h) in rects)
        print(rects)
        pick = non_max_suppression_fast(rects, 0.65)
        
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yB), (xB, yB), (0, 255, 0), 2)
        
        # cv2_imshow(image)
        # cv2.waitKey(0)
        if writer is not None:
            writer.write(image)
        
    else:
        break
if writer is not None:
    writer.release()
    
cv2.destroyAllWindows()

# Malisiewicz et al.
