TF_LITE_MODEL = './lite-model_ssd_mobilenet_v1_1_metadata_2.tflite'
#TF_LITE_MODEL = './lite-model_efficientdet_lite0_detection_metadata_1.tflite'
LABEL_MAP = './labelmap.txt'
THRESHOLD = 0.5
RUNTIME_ONLY = True

IMG_WIDTH = 320
IMG_HEIGHT = 240
LABEL_SIZE = 0.5

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

_, height, width, _ = interpreter.get_input_details()[0]['shape']

with open(LABEL_MAP, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

while cap.isOpened():
    success, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
    for score, box, class_ in zip(scores, boxes, classes):
        if score < THRESHOLD:
            continue
        
        color = [int(c) for c in colors[int(class_)]]
        text_color = (255, 255, 255) if sum(color) < 128 * 3 else (0, 0, 0)
        
        min_y = round(box[0] * IMG_HEIGHT)
        min_x = round(box[1] * IMG_WIDTH)
        max_y = round(box[2] * IMG_HEIGHT)
        max_x = round(box[3] * IMG_WIDTH)
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)
        
        class_name = labels[int(class_)]
        label = f'{class_name}: {score*100:.2f}'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, LABEL_SIZE, 1)
        cv2.rectangle(frame,
                      (min_x, min_y + baseLine), (min_x + labelSize[0], min_y - baseLine - labelSize[1]),
                      color, cv2.FILLED)
        cv2.putText(frame, label, (min_x, min_y), cv2.FONT_HERSHEY_COMPLEX, LABEL_SIZE, text_color, 1)
    
    cv2.imshow('Object Detector', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()