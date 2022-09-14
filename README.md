# Object Detection Examples With Tensorflow Lite and OpenCV (Python)

Running pre-trained TF Lite models for object detection. You either have to install Tehsorflow or Tensorflow Lite (```tflite_runtime```) and OpenCV (```opencv-python```). These scripts also run a lot faster on a ARM device, for example, a Raspberry Pi 3B or 4B.

There are three models available here (downloaded from Google):

* SSD-MobileNet V1
* EfficientDet-Lite0
* YOLO V5

All three are trained with the COCO dataset (```labelmap.txt``` is the label list). This is mainly a demostration of how to get the possible things as well as their location from the model.

![result](https://github.com/alankrantas/TF-Lite-Python-Object-Objection/blob/main/result.jpg)

```TF_Lite_Object_Detection.py``` can use either SSD or EfficientNet to process a still image, and ```TF_Lite_Object_Detection_Yolo.py``` is the YOLO version. ```TF_Lite_Object_Detection_Live.py``` use live USB cam images with SSD or EfficientNet (press ```q```).

