import cv2
import numpy as np
from handDetector.darknet import model
from handDetector.preprocess.yolo_flag import Flag
from handDetector.utils.utils import visualize
from handDetector.detector import YOLO

f = Flag()
model = model() 
model.load_weights('weights/yolo.h5')
image = cv2.imread('data/four.4.jpg', cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (f.target_size, f.target_size))
processed_image = np.expand_dims(image, axis=0) / 255.0
yolo_output = model.predict(processed_image)
yolo_output = yolo_output[0]
visualize(image, yolo_output, title='yolo prediction', RGB2BGR=True)
