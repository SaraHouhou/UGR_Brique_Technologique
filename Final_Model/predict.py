import cv2
import numpy as np
import network
from handDetector.utils.utils import visualize
from handDetector.detector import YOLO

GESTURES_Names = [
    "call",
    "five",
    "four",
    "one",
    "three",
    "two",
    "v"
]
hand = YOLO(weights='handDetector/weights/yolo.h5', threshold=0.8)

model = network.create_model(img_rows=50, img_cols=50, nbclasses=7)
model.load_weights('weights/cp_hand_gesture_cnn_vgg16.h5')
image = cv2.imread('D:/Data/five144.jpg', cv2.COLOR_BGR2RGB)
cv2.imshow('image', image)
tl, br = hand.detect(image=image)

if tl or br is not None:
    cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
    height, width, _ = cropped_image.shape
    # drawing
    index = 0
    color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
    image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (235, 26, 158), 2) 
    # display image
    cv2.imshow(' hand detector', image)
    image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (50, 50))
    cv2.imshow('cropted image and resized', image)
    processed_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    print(processed_image.shape)
    processed_image = np.expand_dims(processed_image, axis=0) / 255.0
    print(processed_image.shape)
    modelPrediction = model.predict(processed_image)
    vgg_output = modelPrediction[0]
    print(vgg_output)
   # visualize(image, vgg_output, title='yolo prediction', RGB2BGR=True)
    maxindex = np.argmax(modelPrediction[0])
    print (GESTURES_Names[maxindex])

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
