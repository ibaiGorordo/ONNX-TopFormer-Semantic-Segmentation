import cv2
import numpy as np
import pafy

from topformer import TopFormer

# Initialize video
# cap = cv2.VideoCapture("input.mp4")

videoUrl = 'https://youtu.be/yWHdkK5j4yk'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)
start_time = 30 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)

# Initialize semantic segmentator
model_path = "models/TopFormer-S_512x512_2x8_160k.onnx"
segmentator = TopFormer(model_path)

cv2.namedWindow("Semantic Sementation", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

	try:
		# Read frame from the video
		ret, frame = cap.read()
		if not ret:	
			break
	except:
		continue
	
	# Update semantic segmentator
	seg_map = segmentator(frame)
	combined_img = segmentator.draw_segmentation(frame, alpha=0.5)
	cv2.imshow("Semantic Sementation", combined_img)
