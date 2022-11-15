import cv2
import numpy as np
import network

GESTURES_Names = [
    "first", 
    "l", 
    "ok", 
    "palm",
    "pointer",
    "thumb_down",
    "thumb_up"
]

model = network.create_model(img_rows=50, img_cols=50, nbclasses=7)
model.load_weights('weights/cp_hand_gesture_cnn_vgg16_TinyHGR.h5')
image = cv2.imread('C:/Users/shouhou/Downloads/TinyHGR/Test/five144.jpg', cv2.COLOR_BGR2RGB)
cv2.imshow('image', image)
image = cv2.resize(image, (224, 224))
cv2.imshow('image resized', image)
processed_image = np.expand_dims(image, axis=0) / 255.0
print(processed_image.shape)
modelPrediction = model.predict(processed_image)
vgg_output = modelPrediction[0]
print(vgg_output)
# visualize(image, vgg_output, title='yolo prediction', RGB2BGR=True)
maxindex = np.argmax(modelPrediction[0])
print (GESTURES_Names[maxindex])


