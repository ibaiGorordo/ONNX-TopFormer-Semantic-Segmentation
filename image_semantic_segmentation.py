import cv2
from imread_from_url import imread_from_url

from topformer import TopFormer

model_path = "models/TopFormer-S_512x512_2x8_160k.onnx"

# Initialize semantic segmentator
segmentator = TopFormer(model_path)

img = imread_from_url("https://upload.wikimedia.org/wikipedia/az/6/6d/Abbey_Road_%28albom%29.jpg")

# Update semantic segmentator
seg_map = segmentator(img)

combined_img = segmentator.draw_segmentation(img, alpha=0.5)
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.imshow("Output", combined_img)
cv2.waitKey(0)
