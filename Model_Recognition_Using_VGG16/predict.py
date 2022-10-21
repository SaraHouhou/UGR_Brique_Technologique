
import os
import numpy as np
from keras.preprocessing import image
from sklearn.metrics import accuracy_score




def predict1(VALIDATION_DIR, MODEL, IMAGE_SIZE):
    images = os.listdir(VALIDATION_DIR)
    for i in images:
        print()
# predicting images
        path = VALIDATION_DIR + i       
        img = image.load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
    
        images = np.vstack([x])
        classes = MODEL.predict(images, batch_size=10)
        print(path)
        print(classes)

def generate_prediction(path, model_history, train_generator, test_Generator):
    # Generate predictions
    model_history.load_weights(path) # initialize the best trained weights

    true_classes = test_Generator.classes
    class_indices = train_generator.class_indices
    class_indices = dict((v,k) for k,v in class_indices.items())
    vgg_preds = model_history.predict(test_Generator)
    vgg_pred_classes = np.argmax(vgg_preds, axis=1)
    vgg_acc = accuracy_score(true_classes, vgg_pred_classes)
    print("VGG16 Model Accuracy without Fine-Tuning: {:.2f}%".format(vgg_acc * 100))


