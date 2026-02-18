from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

def confuse(target, predictions):
    """
    calculates a confusion matrix using target, predictions. returns confusion_matrix(2x2 array), disp(plottable matrix)
    """
    confusion_matrix = metrics.confusion_matrix(target[1],predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['Class 0', 'Class 1']) 
    return confusion_matrix, disp