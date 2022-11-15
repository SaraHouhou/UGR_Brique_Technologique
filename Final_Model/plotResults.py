#plot loss and accuracy

import matplotlib.pyplot as plt

def plotHistory(history_model, acc_title, loss_title):
  acc = history_model.history['accuracy']
  val_acc = history_model.history['val_accuracy']
  loss = history_model.history['loss']
  val_loss = history_model.history['val_loss']

  epochs = range(1, len(acc)+1)

  #plot the train and validation accuracy
  plt.plot(epochs, acc, 'g', label='Training accuracy')
  plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
  plt.title(acc_title)
  plt.legend()
  plt.savefig('C:/Users/shouhou/testcode/UGR_Brique_Technologique/Final_Model/images/accPlot.png')

  plt.figure()

  #plot the train and validation loss
  plt.plot(epochs, loss, 'g', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(loss_title)
  plt.legend()
  plt.savefig('lossPlot.png')
  plt.show()