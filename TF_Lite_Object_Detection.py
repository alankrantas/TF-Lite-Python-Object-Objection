TEST_FILE = './test.jpg'
TF_LITE_MODEL = './lite-model_ssd_mobilenet_v1_1_metadata_2.tflite'
#TF_LITE_MODEL = './lite-model_efficientdet_lite0_detection_metadata_1.tflite'
LABEL_MAP = './labelmap.txt'
THRESHOLD = 0.3
LABEL_SIZE = 1.0
RUNTIME_ONLY = True

import cv2
import numpy as np

if RUNTIME_ONLY:
    from tflite_runtime.interpreter import Interpreter
    interpreter = Interpreter(model_path=TF_LITE_MODEL)
else:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=TF_LITE_MODEL)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

_, INPUT_HEIGHT, INPUT_WIDTH, _ = interpreter.get_input_details()[0]['shape']

with open(LABEL_MAP, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

img = cv2.imread(TEST_FILE, cv2.IMREAD_COLOR)
IMG_HEIGHT, IMG_WIDTH = img.shape[:2]

pad = abs(IMG_WIDTH - IMG_HEIGHT) // 2
x_pad = pad if IMG_HEIGHT > IMG_WIDTH else 0
y_pad = pad if IMG_WIDTH > IMG_HEIGHT else 0
img_padded = cv2.copyMakeBorder(img, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad,
                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
IMG_HEIGHT, IMG_WIDTH = img_padded.shape[:2]

img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
input_data = np.expand_dims(img_resized, axis=0)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

boxes = interpreter.get_tensor(output_details[0]['index'])[0]
classes = interpreter.get_tensor(output_details[1]['index'])[0]
scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
for score, box, class_ in zip(scores, boxes, classes):
    if score < THRESHOLD:
        continue
    
    color = [int(c) for c in colors[int(class_)]]
    text_color = (255, 255, 255) if sum(color) < 144 * 3 else (0, 0, 0)
    
    min_y = round(box[0] * IMG_HEIGHT)
    min_x = round(box[1] * IMG_WIDTH)
    max_y = round(box[2] * IMG_HEIGHT)
    max_x = round(box[3] * IMG_WIDTH)
    cv2.rectangle(img_padded, (min_x, min_y), (max_x, max_y), color, 2)
        
    class_name = labels[int(class_)]
    label = f'{class_name}: {score*100:.2f}%'
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, 1)
    
    cv2.rectangle(img_padded,
                  (min_x, min_y + baseLine), (min_x + labelSize[0], min_y - baseLine - labelSize[1]),
                  color, cv2.FILLED) 
    cv2.putText(img_padded, label, (min_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, text_color, 1)

img_show = img_padded[y_pad: IMG_HEIGHT - y_pad, x_pad: IMG_WIDTH - x_pad]
cv2.namedWindow('Object detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object detection',
                 1024 if IMG_WIDTH > IMG_HEIGHT else round(1024 * IMG_WIDTH / IMG_HEIGHT),
                 1024 if IMG_HEIGHT > IMG_WIDTH else round(1024 * IMG_HEIGHT / IMG_WIDTH))
cv2.imshow('Object detection', img_show)
cv2.imwrite('./result.jpg', img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()