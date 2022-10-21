import matplotlib.pyplot as plt


def diagrams(history):
    #-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(epochs, acc, 'red', label="Training Accuracy")
    plt.plot(epochs, val_acc, 'blue', label="Validation Accuracy")
    plt.legend()
    plt.title('Training and validation accuracy')
   
#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
    plt.subplot(2,1,2)
    plt.plot(epochs, loss, 'bo', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    # Save the image
    plt.savefig('figureTest.png')
    plt.show()
