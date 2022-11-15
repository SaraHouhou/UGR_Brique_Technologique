from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report.csv', index = False)


def classif_report(model, validation_generator, nb_validation_samples, batch_size):
  #Confution Matrix and Classification Report
  Y_pred_CNN = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
  y_pred_CNN = np.argmax(Y_pred_CNN, axis=1)
  print('\n----------------Confusion Matrix-----------------\n')
  print(confusion_matrix(validation_generator.classes, y_pred_CNN))
  class_labels = validation_generator.class_indices
  class_labels = {v: k for k, v in class_labels.items()}
  # Get the names of the six classes
  #class_names = list(class_labels.values())
  target_names = list(class_labels.values())
  print('\n------------- Classification report: Pretrained VGG16 with data augmentation  --------------------\n')
  report=classification_report(validation_generator.classes, y_pred_CNN, target_names=target_names)
  print(report)
  #classification_report_csv(report)
  return y_pred_CNN


  #show the confusion matrix graphically
def plot_heatmap(y_true, y_pred, class_names, ax, title, figpath):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        square=True, 
        xticklabels=class_names, 
        yticklabels=class_names,
        fmt='d', 
        cmap=plt.cm.Blues,
        cbar=False,
        ax=ax
    )
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel('True Label', fontsize=8)
    ax.set_xlabel('Predicted Label', fontsize=8)
    plt.savefig(figpath, dpi=400)

