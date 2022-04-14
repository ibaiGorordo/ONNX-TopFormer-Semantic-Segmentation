import cv2
import numpy as np
import pafy

from topformer import TopFormer

# Initialize video
cap = cv2.VideoCapture(0)

# Initialize semantic segmentator
model_path = "models/TopFormer-S_512x512_2x8_160k.onnx"
segmentator = TopFormer(model_path)

cv2.namedWindow("Semantic Sementation", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	# Read frame from the camera
	ret, frame = cap.read()

	if not ret:	
		break
	
	# Update semantic segmentator
	seg_map = segmentator(frame)
	combined_img = segmentator.draw_segmentation(frame, alpha=0.5)
	cv2.imshow("Semantic Sementation", combined_img)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break
