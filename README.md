# ONNX TopFormer Semantic Segmentation
 Python scripts performing semantic segmentation using the TopFormer model in ONNX.

![!TopFormer Semantic Segmentation](https://github.com/ibaiGorordo/ONNX-TopFormer-Semantic-Segmentation/blob/main/doc/img/output.png)
*Original image: https://en.wikipedia.org/wiki/File:Beatles_-_Abbey_Road.jpg*

# Requirements

 * Check the **requirements.txt** file. 
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.
 * Additionally, **pafy** and **youtube-dl** are required for youtube video inference.
 
# Installation
```
git clone https://github.com/ibaiGorordo/ONNX-TopFormer-Semantic-Segmentation.git
cd ONNX-TopFormer-Semantic-Segmentation
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

### For youtube video inference
```
pip install youtube_dl
pip install git+https://github.com/zizo-pro/pafy@b8976f22c19e4ab5515cacbfae0a3970370c102b
```

# ONNX model
The model was converted from the Pytorch implementation using the [code in the original repository](https://github.com/hustvl/TopFormer/blob/main/tools/convert2onnx.py). Download the converted ONNX model from the [drive file](https://drive.google.com/file/d/1WxvVEqQGn8S2q4uqpG9OZY2Alc5ZcBCa/view?usp=sharing) and save them into the **[models](https://github.com/ibaiGorordo/ONNX-TopFormer-Semantic-Segmentation/tree/main/models)** folder. 
- The License of the models is Apache-2.0 License: https://github.com/hustvl/TopFormer/blob/main/LICENSE

# Pytorch model
The original Pytorch model can be found in this repository: https://github.com/hustvl/TopFormer
 
# Examples

 * **Image inference**:
 ```
 python image_semantic_segmentation.py
 ```
 
 * **Webcam inference**:
 ```
 python webcam_semantic_segmentation.py
 ```

 * **Video inference**: https://youtu.be/JkOSbtKfIFo
 ```
 python video_semantic_segmentation.py
 ```
 ![!CREStereo depth estimation](https://github.com/ibaiGorordo/ONNX-TopFormer-Semantic-Segmentation/blob/main/doc/img/topformer.gif)
  
 *Original video: https://youtu.be/yWHdkK5j4yk*

# References:
* TopFormer model: https://github.com/hustvl/TopFormer
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* Original paper: https://arxiv.org/abs/2204.05525
