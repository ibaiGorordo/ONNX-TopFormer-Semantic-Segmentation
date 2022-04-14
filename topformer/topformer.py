import cv2
import numpy as np
import onnxruntime
from numpy import genfromtxt

from .utils import util_draw_seg

class TopFormer():

	def __init__(self, model_path):

		# Initialize model
		self.initialize_model(model_path)

	def __call__(self, image):
		return self.estimate_segmentation(image)

	def initialize_model(self, model_path):

		self.session = onnxruntime.InferenceSession(model_path, 
													providers=['CUDAExecutionProvider', 
															   'CPUExecutionProvider'])

		# Get model info
		self.get_input_details()
		self.get_output_details()

	def estimate_segmentation(self, image):

		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		outputs = self.inference(input_tensor)

		# Process output data
		self.seg_map = self.process_output(outputs)

		return self.seg_map

	def prepare_input(self, image):

		self.img_height, self.img_width = image.shape[:2]

		input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Resize input image
		input_img = cv2.resize(input_img, (self.input_width,self.input_height))  

		# Scale input pixel values to -1 to 1
		mean=[0.485, 0.456, 0.406]
		std=[0.229, 0.224, 0.225]
		input_img = ((input_img/ 255.0 - mean) / std)
		input_img = input_img.transpose(2, 0, 1)
		input_tensor = input_img[np.newaxis,:,:,:].astype(np.float32)   

		return input_tensor

	def inference(self, input_tensor):

		outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

		return outputs

	def process_output(self, outputs): 

		return np.squeeze(np.argmax(outputs[0], axis=1))

	def draw_segmentation(self, image, alpha = 0.5):

		return util_draw_seg(self.seg_map, image, alpha)

	def get_input_details(self):

		model_inputs = self.session.get_inputs()
		self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

		self.input_shape = model_inputs[0].shape
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def get_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

if __name__ == '__main__':

	from imread_from_url import imread_from_url

	model_path = "../models/TopFormer-S_512x512_2x8_160k.onnx"

	# Initialize semantic segmentator
	segmentator = TopFormer(model_path)

	img = imread_from_url("https://upload.wikimedia.org/wikipedia/az/6/6d/Abbey_Road_%28albom%29.jpg")

	# Update semantic segmentator
	seg_map = segmentator(img)

	combined_img = segmentator.draw_segmentation(img, alpha=0.5)
	cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
	cv2.imshow("Output", combined_img)
	cv2.waitKey(0)
